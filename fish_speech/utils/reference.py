from pathlib import Path

try:
    import pyrootutils
except ImportError:
    pyrootutils = None

DEFAULT_REFERENCE_ID = "default"
DEFAULT_REFERENCE_AUDIO_NAME = "sample.mp3"
DEFAULT_REFERENCE_TEXT_NAME = "sample.lab"
PROJECT_ROOT_INDICATOR = ".project-root"


def get_repository_root() -> Path:
    if pyrootutils is not None:
        try:
            return Path(
                pyrootutils.find_root(
                    search_from=Path(__file__).resolve(),
                    indicator=PROJECT_ROOT_INDICATOR,
                )
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            pass

    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = get_repository_root()
DEFAULT_REFERENCE_AUDIO_PATH = REPOSITORY_ROOT / DEFAULT_REFERENCE_AUDIO_NAME
DEFAULT_REFERENCE_TEXT_PATH = REPOSITORY_ROOT / DEFAULT_REFERENCE_TEXT_NAME


def has_bundled_default_reference() -> bool:
    return DEFAULT_REFERENCE_AUDIO_PATH.exists() and DEFAULT_REFERENCE_TEXT_PATH.exists()
