"""AutoRedTeam public package metadata."""

from importlib import metadata as importlib_metadata
from pathlib import Path
import re


def _distribution_version() -> str:
    try:
        return importlib_metadata.version("glacis-autoredteam")
    except importlib_metadata.PackageNotFoundError:
        # Source checkouts are useful without installation. Derive the fallback
        # from the package manifest instead of maintaining another version copy.
        try:
            pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
            match = re.search(
                r'^version\s*=\s*"([^"]+)"\s*$',
                pyproject.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        except OSError:
            match = None
        return match.group(1) if match else "0+unknown"


__version__ = _distribution_version()

__all__ = ["__version__"]
