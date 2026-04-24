"""Thread-safe manifest writers for downloader audit files."""

import csv
import json
import threading
from pathlib import Path

MANIFEST_LOCK = threading.Lock()


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSON Lines manifest in a thread-safe way."""
    if not records:
        return

    with MANIFEST_LOCK:
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_species_summary(path: Path, row: dict) -> None:
    """Append one species-level summary row to a TSV file."""
    fieldnames = [
        "run_id",
        "species_name",
        "canonical_name",
        "taxon_id",
        "candidates",
        "scanned_candidates",
        "downloaded",
        "download_failed",
        "accepted",
        "rejected",
        "unused_valid",
        "search_exhausted",
    ]

    with MANIFEST_LOCK:
        should_write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if should_write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})
