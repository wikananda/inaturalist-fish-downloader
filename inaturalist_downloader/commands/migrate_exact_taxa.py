"""Build a corrected exact-taxon snapshot from an accepted V3 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from ..common.utils import slugify


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "manifests/broad_coral_global_v3/accepted.jsonl"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT
DEFAULT_PLAN = PROJECT_ROOT / "plans/broad_coral_global_v4/broad_train_species.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "broad_coral_global_v4_primary_downloads"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "manifests/broad_coral_global_v4/accepted.jsonl"


def _load_plan(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["taxon_id"]): {
                "taxon_id": int(row["taxon_id"]),
                "species": row["species"],
                "target": int(row["target"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        }


def migrate(
    source_manifest: Path,
    source_root: Path,
    plan_path: Path,
    output_dir: Path,
    output_manifest: Path,
    *,
    copy_files: bool = False,
    merge_existing: bool = False,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    rows: list[dict[str, Any]] = []
    manifest_keys: set[tuple[int, int]] = set()
    linked = Counter()
    skipped = Counter()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge_existing and output_manifest.exists():
        with output_manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                output_path = record.get("saved_output_path")
                observation_id = record.get("observation_id")
                photo_id = record.get("photo_id")
                if (
                    output_path
                    and Path(output_path).is_file()
                    and observation_id is not None
                    and photo_id is not None
                ):
                    rows.append(record)
                    manifest_keys.add((int(observation_id), int(photo_id)))

    def destination_for(source: Path, species: str, observation_id: Any, photo_id: Any) -> Path:
        species_slug = slugify(species)
        suffix = source.suffix.lower() or ".jpg"
        filename = (
            f"{species_slug}__obs_{observation_id}__photo_{photo_id}{suffix}"
        )
        return output_dir / species_slug / filename

    def same_content(left: Path, right: Path) -> bool:
        if left.stat().st_size != right.stat().st_size:
            return False
        left_hash = hashlib.sha256(left.read_bytes()).digest()
        right_hash = hashlib.sha256(right.read_bytes()).digest()
        return left_hash == right_hash

    with source_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            observed_taxon_id = record.get("observation_taxon_id")
            target = plan.get(int(observed_taxon_id or 0))
            if target is None:
                observed_rank = str(record.get("observation_taxon_rank") or "").casefold()
                if observed_rank in {"subspecies", "variety", "form"}:
                    ancestor_ids = {
                        int(value)
                        for value in record.get("observation_ancestor_ids") or []
                    }
                    matching_ancestors = ancestor_ids.intersection(plan)
                    if len(matching_ancestors) == 1:
                        target = plan[matching_ancestors.pop()]
            if target is None:
                skipped["taxon_not_in_migration_plan"] += 1
                continue
            source_value = record.get("saved_output_path")
            if not source_value:
                skipped["missing_source_path"] += 1
                continue
            source = Path(source_value)
            if not source.is_absolute():
                source = source_root / source
            if not source.is_file():
                skipped["missing_source_file"] += 1
                continue

            manifest_key = (
                int(record.get("observation_id")),
                int(record.get("photo_id")),
            )
            if manifest_key in manifest_keys:
                skipped["already_in_output_manifest"] += 1
                continue

            destination = destination_for(
                source,
                target["species"],
                record.get("observation_id"),
                record.get("photo_id"),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not same_content(source, destination):
                raise FileExistsError(
                    f"Migration destination exists with different content: {destination}"
                )
            if not destination.exists():
                if copy_files:
                    shutil.copy2(source, destination)
                else:
                    os.link(source, destination)
            migrated = {
                **record,
                "migration_source_canonical_name": record.get("canonical_name"),
                "migration_source_taxon_id": record.get("taxon_id"),
                "species_name": target["species"],
                "canonical_name": target["species"],
                "taxon_id": target["taxon_id"],
                "training_taxon_id": target["taxon_id"],
                "requested_taxon_id": target["taxon_id"],
                "filename": destination.name,
                "target_output_path": str(destination),
                "saved_output_path": str(destination),
                "output_path_exists": True,
                "migration_method": "copy" if copy_files else "hardlink",
            }
            rows.append(migrated)
            manifest_keys.add(manifest_key)
            linked[target["species"]] += 1

    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_manifest)
    report = {
        "source_manifest": str(source_manifest),
        "plan": str(plan_path),
        "output_dir": str(output_dir),
        "output_manifest": str(output_manifest),
        "migration_method": "copy" if copy_files else "hardlink",
        "merge_existing": merge_existing,
        "retained_existing_records": len(rows) - sum(linked.values()),
        "migrated_records": len(rows),
        "newly_migrated_records": sum(linked.values()),
        "migrated_species": len(linked),
        "migrated_by_species": dict(sorted(linked.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    (output_manifest.parent / "migration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-link or copy V3 crops into corrected exact-species folders."
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking files.")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Retain valid rows already in the output manifest (for reserve top-up).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            migrate(
                args.source_manifest,
                args.source_root,
                args.plan,
                args.output_dir,
                args.output_manifest,
                copy_files=args.copy,
                merge_existing=args.merge_existing,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
