import re

LANGUAGE_HINT_ALIASES = {
    "auto": None,
    "default": None,
    "spanish": "es",
    "es-es": "es",
    "es_mx": "es",
    "es-mx": "es",
    "español": "es",
    "castellano": "es",
    "english": "en",
    "en-us": "en",
    "en_us": "en",
    "en-gb": "en",
    "en_gb": "en",
}

SPANISH_CHARACTER_PATTERN = re.compile(r"[áéíóúüñÁÉÍÓÚÜÑ¿¡]")
SPANISH_KEYWORD_PATTERN = re.compile(
    r"\b(?:hola|gracias|adios|adiós|por|para|como|cómo|esta|está|estoy|estamos|"
    r"señor|señora|señores|texto|voz|audio|ejemplo|medir|factor|tiempo|real|"
    r"rendimiento|latencia|mañana|niño|niña|buenos|buenas|dias|días|tardes|"
    r"noches|mundo|usted|ustedes|nosotros|vosotros|quiero|puedo|favor)\b",
    re.IGNORECASE,
)
SPANISH_FUNCTION_WORD_PATTERN = re.compile(
    r"\b(?:el|la|los|las|un|una|unos|unas|de|del|que|en|es|y|con|sin|por|para)\b",
    re.IGNORECASE,
)


def canonicalize_language_hint(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = language.strip().lower()
    if not normalized:
        return None

    return LANGUAGE_HINT_ALIASES.get(normalized, normalized)


def looks_like_spanish(text: str) -> bool:
    if not text:
        return False

    if SPANISH_CHARACTER_PATTERN.search(text):
        return True

    keyword_hits = len(SPANISH_KEYWORD_PATTERN.findall(text))
    function_hits = len(SPANISH_FUNCTION_WORD_PATTERN.findall(text))
    return keyword_hits >= 2 or (keyword_hits >= 1 and function_hits >= 2)


def resolve_tts_language(text: str, requested_language: str | None) -> str | None:
    resolved = canonicalize_language_hint(requested_language)
    if looks_like_spanish(text):
        return "es"
    return resolved
