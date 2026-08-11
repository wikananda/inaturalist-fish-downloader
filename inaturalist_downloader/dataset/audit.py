"""Snapshot and diff helpers for manual image curation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import IMAGE_EXTENSIONS

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a project dependency.
    Image = None


SNAPSHOT_VERSION = 1


def iter_image_paths(images_dir: Path) -> Iterable[Path]:
    """Yield image files under an accepted-image directory."""
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_snapshot(images_dir: Path) -> dict[str, Any]:
    """Build a stable metadata snapshot for all images under images_dir."""
    images_dir = images_dir.resolve()
    files = {}
    for path in iter_image_paths(images_dir):
        relative_path = path.relative_to(images_dir).as_posix()
        files[relative_path] = image_record(path, images_dir)

    return {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(images_dir),
        "files": files,
    }


def image_record(path: Path, images_dir: Path) -> dict[str, Any]:
    """Return metadata used to detect removed or edited images."""
    stat = path.stat()
    width = None
    height = None
    image_format = None
    if Image is not None:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
        except Exception:
            pass

    relative_path = path.relative_to(images_dir).as_posix()
    parts = Path(relative_path).parts
    species_slug = parts[0] if len(parts) > 1 else ""
    return {
        "path": relative_path,
        "species_slug": species_slug,
        "filename": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "format": image_format,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    """Write a snapshot JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load and validate an image audit snapshot."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("version") != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported snapshot version: {snapshot.get('version')}")
    if not isinstance(snapshot.get("files"), dict):
        raise ValueError("Snapshot is missing a files mapping")
    return snapshot


def diff_snapshot(snapshot: dict[str, Any], images_dir: Path) -> list[dict[str, Any]]:
    """Compare a snapshot with the current directory state."""
    before_files = snapshot["files"]
    after_files = build_snapshot(images_dir)["files"]
    changes = []

    for path in sorted(set(before_files).union(after_files)):
        before = before_files.get(path)
        after = after_files.get(path)
        if before is None:
            changes.append({"status": "added", "path": path, "before": None, "after": after})
            continue
        if after is None:
            changes.append(
                {"status": "deleted", "path": path, "before": before, "after": None}
            )
            continue

        change_types = changed_fields(before, after)
        if change_types:
            changes.append(
                {
                    "status": "modified",
                    "path": path,
                    "change_types": change_types,
                    "before": before,
                    "after": after,
                }
            )

    return changes


def changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Return meaningful metadata fields changed by manual edits."""
    fields = {
        "sha256": "content",
        "size_bytes": "size",
        "width": "dimensions",
        "height": "dimensions",
        "format": "format",
    }
    changes = []
    for field, label in fields.items():
        if before.get(field) != after.get(field) and label not in changes:
            changes.append(label)
    return changes


def write_changes_jsonl(changes: list[dict[str, Any]], path: Path) -> None:
    """Write change records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for change in changes:
            file.write(json.dumps(change, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_changes(changes: list[dict[str, Any]]) -> dict[str, int]:
    """Count audit changes by status."""
    summary = {"added": 0, "deleted": 0, "modified": 0}
    for change in changes:
        status = change.get("status")
        if status in summary:
            summary[status] += 1
    summary["total"] = sum(summary.values())
    return summary
