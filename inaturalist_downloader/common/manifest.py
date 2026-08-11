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
    """Atomically upsert the latest species-level summary row to a TSV file."""
    fieldnames = [
        "run_id",
        "species_name",
        "canonical_name",
        "taxon_id",
        "target_unit",
        "target",
        "candidates",
        "scanned_candidates",
        "downloaded",
        "download_failed",
        "accepted",
        "accepted_outputs",
        "accepted_observations",
        "rejected",
        "unused_valid",
        "search_exhausted",
        "stop_reason",
    ]

    normalized = {key: row.get(key, "") for key in fieldnames}
    identity = str(normalized.get("taxon_id") or normalized.get("canonical_name"))

    with MANIFEST_LOCK:
        rows = []
        replaced = False
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                for existing in csv.DictReader(handle, delimiter="\t"):
                    existing_identity = str(
                        existing.get("taxon_id") or existing.get("canonical_name")
                    )
                    if existing_identity == identity:
                        if not replaced:
                            rows.append(normalized)
                            replaced = True
                        continue
                    rows.append(existing)
        if not replaced:
            rows.append(normalized)

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for existing in rows:
                writer.writerow(
                    {key: existing.get(key, "") for key in fieldnames}
                )
        temporary.replace(path)
