"""Dataset checking helpers."""

from pathlib import Path

from .config import IMAGE_EXTENSIONS


def count_images(folder: Path) -> int:
    """Count image files directly inside one species folder."""
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def species_name_from_folder_slug(value: str) -> str:
    """Convert a downloader species folder slug back into a taxon query string."""
    return " ".join(part for part in value.strip().split("_") if part)


def load_species_set(path: Path) -> set[str]:
    """Load a newline-delimited species file as a set."""
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
