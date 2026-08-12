"""General utility helpers used by the downloader."""

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class SpeciesRequest:
    """One exact downloader request from a text or TSV species plan."""

    species: str
    taxon_id: int | None = None
    target: int | None = None


def _optional_positive_int(value: str, *, field: str, path: Path, line: int) -> int | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{path}:{line}: {field} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{path}:{line}: {field} must be greater than 0")
    return parsed


def load_species_requests(path: Path) -> list[SpeciesRequest]:
    """Load name-only, TSV, or CSV species requests.

    Structured files must contain ``species`` (or ``canonical_name``) and may
    contain ``taxon_id`` plus ``target``/``planned_accepted_target``. Exact IDs
    prevent ambiguous iNaturalist autocomplete results such as species complexes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Species file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    content = [(index, raw) for index, raw in enumerate(lines, start=1) if raw.strip()]
    if not content:
        return []

    first_line = content[0][1]
    delimiter = "\t" if "\t" in first_line else "," if "," in first_line else None
    header = [part.strip().casefold() for part in first_line.split(delimiter)] if delimiter else []
    name_field = next(
        (field for field in ("species", "canonical_name", "species_name") if field in header),
        None,
    )
    if delimiter and name_field:
        import csv

        reader = csv.DictReader(lines, delimiter=delimiter)
        requests = []
        for line_number, row in enumerate(reader, start=2):
            species = str(row.get(name_field) or "").strip()
            if not species or species.startswith("#"):
                continue
            taxon_id = _optional_positive_int(
                row.get("taxon_id", ""), field="taxon_id", path=path, line=line_number
            )
            target_value = row.get("target", "") or row.get("planned_accepted_target", "")
            target = _optional_positive_int(
                target_value, field="target", path=path, line=line_number
            )
            requests.append(SpeciesRequest(species, taxon_id, target))
        return requests

    requests = []
    for line_number, raw_line in content:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Compact exact format: ``taxon_id<TAB>species<TAB>target``.
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) >= 2 and parts[0].isdigit():
            requests.append(
                SpeciesRequest(
                    species=parts[1],
                    taxon_id=_optional_positive_int(
                        parts[0], field="taxon_id", path=path, line=line_number
                    ),
                    target=_optional_positive_int(
                        parts[2] if len(parts) >= 3 else "",
                        field="target",
                        path=path,
                        line=line_number,
                    ),
                )
            )
        else:
            requests.append(SpeciesRequest(species=line))
    return requests


def safe_print(message: str) -> None:
    """Print a message without interleaving output from worker threads."""
    with PRINT_LOCK:
        print(message)


def load_species(path: Path) -> list[str]:
    """Load species names while retaining compatibility with structured plans."""
    return [request.species for request in load_species_requests(path)]


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
