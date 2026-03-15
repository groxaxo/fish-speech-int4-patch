import math
import re
from functools import lru_cache

import inflect
from pydantic import BaseModel, Field

from .clean import clean_text

VALID_TLDS = [
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "io",
    "co",
    "us",
    "uk",
    "de",
    "fr",
    "jp",
]

VALID_UNITS = {
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "m": "meter",
    "kg": "kilogram",
    "g": "gram",
    "ms": "millisecond",
    "min": "minute",
    "h": "hour",
    "s": "second",
    "mph": "mile per hour",
    "km/h": "kilometer per hour",
    "kph": "kilometer per hour",
    "gb": "gigabyte",
    "mb": "megabyte",
    "kb": "kilobyte",
    "tb": "terabyte",
    "°c": "degree celsius",
    "°f": "degree fahrenheit",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
}

SYMBOL_REPLACEMENTS = {
    "~": " ",
    "@": " at ",
    "#": " number ",
    "$": " dollar ",
    "%": " percent ",
    "^": " ",
    "&": " and ",
    "*": " ",
    "_": " ",
    "|": " ",
    "\\": " ",
    "/": " slash ",
    "=": " equals ",
    "+": " plus ",
}

MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)
URL_PATTERN = re.compile(
    r"(https?://|www\.)?(localhost|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|[0-9]{1,3}(?:\.[0-9]{1,3}){3})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(VALID_UNITS.keys(), reverse=True))
    + r")(?=[^\w\d]|\b)",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"([0-9]{1,2}\s*:\s*[0-9]{2}(?:\s*:\s*[0-9]{2})?)(\s*(pm|am)\b)?",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
)


class TextNormalizationOptions(BaseModel):
    normalize: bool = Field(
        default=True,
        description="Normalize input text to improve pronunciation and stability.",
    )
    unit_normalization: bool = Field(
        default=False,
        description="Expand units like 10km or 128GB into pronounceable text.",
    )
    url_normalization: bool = Field(
        default=True,
        description="Expand URLs into pronounceable text.",
    )
    email_normalization: bool = Field(
        default=True,
        description="Expand email addresses into pronounceable text.",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replace optional plural markers like (s).",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Expand phone numbers into pronounceable text.",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replace symbols like %, + and = after normalization.",
    )


@lru_cache(maxsize=1)
def _inflect_engine():
    return inflect.engine()


def _conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def _translate_multiplier(multiplier: str) -> str:
    return {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }.get(multiplier.lower(), multiplier.strip())


def _split_four_digit(number: float) -> str:
    engine = _inflect_engine()
    whole = str(_conditional_int(number))
    return (
        f"{engine.number_to_words(whole[:2])} "
        f"{engine.number_to_words(whole[2:])}"
    )


def _handle_units(match: re.Match[str]) -> str:
    engine = _inflect_engine()
    unit_string = match.group(6).strip()
    spoken_unit = VALID_UNITS.get(unit_string.lower(), unit_string)
    count = match.group(1).strip()
    return engine.no(spoken_unit, count)


def _handle_numbers(match: re.Match[str]) -> str:
    engine = _inflect_engine()
    number = match.group(2)
    try:
        value = float(number)
    except ValueError:
        return match.group()

    if match.group(1) == "-":
        value *= -1

    multiplier = _translate_multiplier(match.group(3))
    value = _conditional_int(value)
    if multiplier:
        return f"{engine.number_to_words(value)} {multiplier}"

    if (
        isinstance(value, int)
        and len(str(value)) == 4
        and value > 1500
        and value % 1000 > 9
    ):
        return _split_four_digit(value)

    return engine.number_to_words(value)


def _handle_money(match: re.Match[str]) -> str:
    engine = _inflect_engine()
    bill, coin = MONEY_UNITS[match.group(2)]
    try:
        value = float(match.group(3))
    except ValueError:
        return match.group()

    if match.group(1) == "-":
        value *= -1

    multiplier = _translate_multiplier(match.group(4))
    if multiplier:
        return (
            f"{engine.number_to_words(_conditional_int(value))} "
            f"{multiplier} {engine.plural(bill, count=value)}"
        )

    if value % 1 == 0:
        return (
            f"{engine.number_to_words(_conditional_int(value))} "
            f"{engine.plural(bill, count=value)}"
        )

    cents = int(str(value).split(".")[-1].ljust(2, "0"))
    return (
        f"{engine.number_to_words(int(math.floor(value)))} "
        f"{engine.plural(bill, count=value)} and "
        f"{engine.number_to_words(cents)} {engine.plural(coin, count=cents)}"
    )


def _handle_decimal(match: re.Match[str]) -> str:
    whole, fractional = match.group().split(".")
    return " point ".join([whole, " ".join(fractional)])


def _handle_email(match: re.Match[str]) -> str:
    user, domain = match.group(0).split("@", 1)
    return f"{user} at {domain.replace('.', ' dot ')}"


def _handle_url(match: re.Match[str]) -> str:
    url = match.group(0).strip()
    if not url:
        return ""

    url = re.sub(
        r"^https?://",
        lambda value: "https " if "https" in value.group().lower() else "http ",
        url,
        flags=re.IGNORECASE,
    )
    url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)
    url = re.sub(r":(\d+)(?=/|$)", lambda value: f" colon {value.group(1)}", url)

    parts = url.split("/", 1)
    domain = parts[0].replace(".", " dot ")
    path = parts[1] if len(parts) > 1 else ""
    rebuilt = f"{domain} slash {path}" if path else domain

    return re.sub(
        r"\s+",
        " ",
        rebuilt.replace("-", " dash ")
        .replace("_", " underscore ")
        .replace("?", " question-mark ")
        .replace("=", " equals ")
        .replace("&", " ampersand ")
        .replace("%", " percent ")
        .replace(":", " colon ")
        .replace("/", " slash "),
    ).strip()


def _handle_phone_number(match: re.Match[str]) -> str:
    engine = _inflect_engine()
    country_code, _, area_code, telephone_prefix, line_number = match.groups()
    spoken_parts = []
    if country_code:
        spoken_parts.append(
            engine.number_to_words(
                country_code.replace("+", ""), group=1, comma=""
            ).replace(",", "")
        )
    spoken_parts.append(
        engine.number_to_words(
            area_code.replace("(", "").replace(")", ""), group=1, comma=""
        ).replace(",", "")
    )
    spoken_parts.append(
        engine.number_to_words(telephone_prefix, group=1, comma="").replace(",", "")
    )
    spoken_parts.append(
        engine.number_to_words(line_number, group=1, comma="").replace(",", "")
    )
    separator = " " if match.group(2) else ""
    return separator + ", ".join(spoken_parts)


def _handle_time(match: re.Match[str]) -> str:
    engine = _inflect_engine()
    time_value, _, meridiem = match.groups()
    parts = [part.strip() for part in time_value.split(":")]
    numbers = [engine.number_to_words(parts[0])]

    minute = int(parts[1])
    minute_spoken = engine.number_to_words(parts[1])
    if minute == 0 and len(parts) == 2 and meridiem is None:
        numbers.append("o'clock")
    elif minute < 10:
        numbers.append(f"oh {minute_spoken}")
    else:
        numbers.append(minute_spoken)

    if len(parts) > 2:
        seconds = engine.number_to_words(parts[2])
        numbers.append(f"and {seconds} {engine.plural('second', int(parts[2]))}")

    if meridiem:
        numbers.append(meridiem.strip())

    return " ".join(numbers)


def merge_normalization_options(
    normalize: bool,
    normalization_options: TextNormalizationOptions | None = None,
) -> TextNormalizationOptions:
    options = (
        normalization_options.model_copy(deep=True)
        if normalization_options is not None
        else TextNormalizationOptions()
    )

    if not normalize:
        return TextNormalizationOptions(
            normalize=False,
            unit_normalization=False,
            url_normalization=False,
            email_normalization=False,
            optional_pluralization_normalization=False,
            phone_normalization=False,
            replace_remaining_symbols=False,
        )

    options.normalize = True
    return options


def normalize_text_for_tts(
    text: str,
    normalize: bool = True,
    normalization_options: TextNormalizationOptions | None = None,
) -> str:
    text = clean_text(text or "")
    if not text:
        return ""

    options = merge_normalization_options(normalize, normalization_options)
    if not options.normalize:
        return re.sub(r"\s{2,}", " ", text.replace("\n", " ").replace("\r", " ")).strip()

    if options.email_normalization:
        text = EMAIL_PATTERN.sub(_handle_email, text)

    if options.url_normalization:
        text = URL_PATTERN.sub(_handle_url, text)

    if options.unit_normalization:
        text = UNIT_PATTERN.sub(_handle_units, text)

    if options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)

    if options.phone_normalization:
        text = PHONE_PATTERN.sub(_handle_phone_number, text)

    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')

    for source, target in zip("、。！，：；？", ",.!,:;?"):
        text = text.replace(source, target + " ")
    text = text.replace("–", "- ")

    text = TIME_PATTERN.sub(_handle_time, text)

    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)
    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)

    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = MONEY_PATTERN.sub(_handle_money, text)
    text = NUMBER_PATTERN.sub(_handle_numbers, text)
    text = re.sub(r"\d*\.\d+", _handle_decimal, text)

    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", lambda value: value.group().replace(".", "-"), text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

    return re.sub(r"\s{2,}", " ", text).strip()
