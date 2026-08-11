"""Build a filtered dataset from Label Studio validation CSV exports."""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse


VALIDATED_VALUE = "validated"
DEFAULT_VALID_COLUMN = "valid"
DEFAULT_IMAGE_COLUMN = "image"
DEFAULT_LABEL_COLUMN = "label"


def read_label_studio_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read Label Studio CSV rows with UTF-8 BOM tolerance."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def extract_validated_images(
    *,
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    report_path: Path,
    image_column: str = DEFAULT_IMAGE_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    valid_column: str = DEFAULT_VALID_COLUMN,
    valid_value: str = VALIDATED_VALUE,
    mode: str = "copy",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy/link validated images into class folders and write an audit report."""
    rows = read_label_studio_rows(csv_path)
    changes = []
    seen_outputs: set[str] = set()

    for index, row in enumerate(rows, start=1):
        change = process_validation_row(
            row=row,
            row_number=index,
            images_dir=images_dir,
            output_dir=output_dir,
            image_column=image_column,
            label_column=label_column,
            valid_column=valid_column,
            valid_value=valid_value,
            mode=mode,
            overwrite=overwrite,
            dry_run=dry_run,
            seen_outputs=seen_outputs,
        )
        changes.append(change)

    write_report(changes, report_path)
    return summarize_report(changes)


def process_validation_row(
    *,
    row: dict[str, str],
    row_number: int,
    images_dir: Path,
    output_dir: Path,
    image_column: str,
    label_column: str,
    valid_column: str,
    valid_value: str,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    seen_outputs: set[str],
) -> dict[str, Any]:
    """Process one Label Studio CSV row."""
    label = (row.get(label_column) or "").strip()
    image_value = (row.get(image_column) or "").strip()
    valid = (row.get(valid_column) or "").strip()
    record = {
        "row_number": row_number,
        "status": None,
        "label": label,
        "valid": valid,
        "source": None,
        "output": None,
        "image": image_value,
    }

    if valid.casefold() != valid_value.casefold():
        record["status"] = "ignored_unvalidated"
        return record
    if not label:
        record["status"] = "missing_label"
        return record
    if not image_value:
        record["status"] = "missing_image"
        return record

    source = resolve_source_path(image_value, label=label, images_dir=images_dir)
    record["source"] = str(source)
    if not source.exists():
        record["status"] = "missing_source"
        return record

    destination = output_dir / label / source.name
    record["output"] = str(destination)
    output_key = destination.as_posix()
    if output_key in seen_outputs:
        record["status"] = "duplicate"
        return record
    seen_outputs.add(output_key)

    if destination.exists() and not overwrite:
        record["status"] = "skipped_existing"
        return record

    record["status"] = mode_status(mode, dry_run=dry_run)
    if not dry_run:
        materialize_image(source, destination, mode=mode, overwrite=overwrite)
    return record


def resolve_source_path(image_value: str, *, label: str, images_dir: Path) -> Path:
    """Resolve Label Studio image references to local downloads paths."""
    parsed = urlparse(image_value)
    candidates = []

    query_path = parse_qs(parsed.query).get("d", [""])[0]
    if query_path:
        candidates.append(path_from_downloads_fragment(query_path, images_dir))

    if parsed.path:
        candidates.append(path_from_downloads_fragment(parsed.path, images_dir))
        candidates.append(images_dir / label / Path(unquote(parsed.path)).name)

    raw_path = Path(unquote(image_value))
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(images_dir / raw_path)
        candidates.append(images_dir / label / raw_path.name)

    for candidate in unique_paths(candidates):
        if candidate.exists():
            return candidate
    return unique_paths(candidates)[0]


def path_from_downloads_fragment(value: str, images_dir: Path) -> Path:
    """Return images_dir-relative path from a value containing a downloads segment."""
    path = Path(unquote(value))
    parts = path.parts
    if "downloads" in parts:
        index = parts.index("downloads")
        return images_dir / Path(*parts[index + 1 :])
    return images_dir / path.name


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Deduplicate paths while preserving order."""
    seen = set()
    unique = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def materialize_image(
    source: Path,
    destination: Path,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    """Create the filtered dataset file using the requested mode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and overwrite:
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        destination.hardlink_to(source)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def mode_status(mode: str, *, dry_run: bool) -> str:
    """Return report status for a materialization mode."""
    if dry_run:
        return f"would_{mode}"
    return {"copy": "copied", "symlink": "symlinked", "hardlink": "hardlinked"}[mode]


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    """Write extraction records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_report(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count extraction records by status."""
    summary: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        summary[status] = summary.get(status, 0) + 1
    summary["total"] = len(records)
    return summary


def status_count(summary: dict[str, int], statuses: Iterable[str]) -> int:
    """Return the combined count for multiple statuses."""
    return sum(summary.get(status, 0) for status in statuses)
