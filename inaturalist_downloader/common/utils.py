"""General utility helpers used by the downloader."""

import re
import threading
from pathlib import Path
from typing import Optional

PRINT_LOCK = threading.Lock()


def safe_print(message: str) -> None:
    """Print a message without interleaving output from worker threads."""
    with PRINT_LOCK:
        print(message)


def load_species(path: Path) -> list[str]:
    """Load species names from a newline-delimited text file."""
    if not path.exists():
        raise FileNotFoundError(f"Species file not found: {path}")

    species = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        species.append(line)
    return species


def slugify(value: str) -> str:
    """Convert a species/taxon name into a filesystem-safe folder slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "species"


def parse_csv_set(value: Optional[str]) -> set[str]:
    """Parse a comma-separated CLI value into a normalized string set."""
    if not value:
        return set()
    return {item.strip().casefold() for item in value.split(",") if item.strip()}


def parse_csv_int_set(value: Optional[str]) -> set[int]:
    """Parse comma-separated integer IDs from a CLI value."""
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}
