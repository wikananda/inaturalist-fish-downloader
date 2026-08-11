"""Summarize automatic crop-filter outcomes from downloader manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _species_name(record: dict[str, Any]) -> str:
    return str(
        record.get("canonical_name") or record.get("species_name") or "unknown"
    )


def _duplicate_summary(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value not in (None, ""):
            groups[str(value)].append(record)
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {
        "groups": len(duplicates),
        "records": sum(len(group) for group in duplicates),
        "cross_species_groups": sum(
            len({_species_name(record).casefold() for record in group}) > 1
            for group in duplicates
        ),
    }


def _read_species_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _resolve_accepted_path(record: dict[str, Any], manifest_dir: Path) -> Path | None:
    value = record.get("saved_output_path") or record.get("target_output_path")
    if not value:
        return None
    raw_path = Path(str(value))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend(
            [manifest_dir.parent.parent / raw_path, manifest_dir.parent / raw_path]
        )
    return next((path for path in candidates if path.is_file()), None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "count": len(values),
        "min": round(ordered[0], 6),
        "p10": round(percentile(0.10), 6),
        "median": round(float(statistics.median(ordered)), 6),
        "p90": round(percentile(0.90), 6),
        "max": round(ordered[-1], 6),
    }


def build_quality_report(manifest_dir: Path) -> dict[str, Any]:
    """Build acceptance, rejection, and score summaries from one manifest run."""
    root = Path(manifest_dir)
    candidates = list(_read_jsonl(root / "candidates.jsonl"))
    accepted = list(_read_jsonl(root / "accepted.jsonl"))
    rejected = list(_read_jsonl(root / "rejected.jsonl"))
    failures = list(_read_jsonl(root / "failures.jsonl"))
    summary_rows = _read_species_summary(root / "species_summary.tsv")
    legacy_hashes_computed = 0
    accepted_paths_unavailable_for_hashing = 0
    for record in accepted:
        if record.get("content_sha256"):
            continue
        output_path = _resolve_accepted_path(record, root)
        if output_path is None:
            accepted_paths_unavailable_for_hashing += 1
            continue
        try:
            record["content_sha256"] = _sha256_file(output_path)
            legacy_hashes_computed += 1
        except OSError:
            accepted_paths_unavailable_for_hashing += 1
    reject_reasons = Counter(
        str(record.get("reject_reason") or "unknown") for record in rejected
    )
    accepted_by_species = Counter(_species_name(record) for record in accepted)
    rejected_by_species = Counter(_species_name(record) for record in rejected)
    semantic_scores: dict[str, list[float]] = defaultdict(list)
    crop_edge_variances: dict[str, list[float]] = defaultdict(list)

    for outcome, records in (("accepted", accepted), ("rejected", rejected)):
        for record in records:
            clip = record.get("clip") or {}
            if clip.get("context_score") is not None:
                semantic_scores[outcome].append(float(clip["context_score"]))
            crop_quality = (record.get("detection") or {}).get("crop_quality") or {}
            visual = crop_quality.get("visual") or {}
            if visual.get("edge_variance") is not None:
                crop_edge_variances[outcome].append(float(visual["edge_variance"]))

    species_rows = []
    for species in sorted(set(accepted_by_species) | set(rejected_by_species)):
        accepted_count = accepted_by_species[species]
        rejected_count = rejected_by_species[species]
        decided = accepted_count + rejected_count
        species_rows.append(
            {
                "species": species,
                "accepted": accepted_count,
                "rejected": rejected_count,
                "acceptance_rate": (
                    round(accepted_count / decided, 6) if decided else None
                ),
            }
        )

    decided_count = len(accepted) + len(rejected)
    accepted_license_counts = Counter(
        str(record.get("license_code") or "missing").lower()
        for record in accepted
    )
    allowed_commercial_derivative_licenses = {"cc0", "cc-by", "cc-by-sa"}
    unsafe_license_records = sum(
        str(record.get("license_code") or "").lower()
        not in allowed_commercial_derivative_licenses
        for record in accepted
    )
    stop_reasons = Counter(
        str(row.get("stop_reason") or "missing") for row in summary_rows
    )
    return {
        "manifest_dir": str(root),
        "candidate_records": len(candidates),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "acceptance_rate": (
            round(len(accepted) / decided_count, 6) if decided_count else None
        ),
        "reject_reasons": dict(reject_reasons.most_common()),
        "dataset_safety": {
            "accepted_license_counts": dict(accepted_license_counts.most_common()),
            "accepted_unsafe_or_missing_license_records": unsafe_license_records,
            "accepted_duplicate_observation_ids": _duplicate_summary(
                accepted, "observation_id"
            ),
            "accepted_duplicate_photo_ids": _duplicate_summary(accepted, "photo_id"),
            "accepted_duplicate_exact_content": _duplicate_summary(
                accepted, "content_sha256"
            ),
            "legacy_content_hashes_computed": legacy_hashes_computed,
            "accepted_paths_unavailable_for_hashing": (
                accepted_paths_unavailable_for_hashing
            ),
            "rejected_duplicate_or_label_conflicts": sum(
                reason.startswith("duplicate_") or reason.startswith("conflicting_")
                for reason in (
                    str(record.get("reject_reason") or "") for record in rejected
                )
            ),
        },
        "run_completion": {
            "species_summary_rows": len(summary_rows),
            "stop_reasons": dict(stop_reasons.most_common()),
            "failed_records": len(failures),
        },
        "semantic_context_scores": {
            outcome: _score_summary(values)
            for outcome, values in sorted(semantic_scores.items())
        },
        "crop_edge_variance": {
            outcome: _score_summary(values)
            for outcome, values in sorted(crop_edge_variances.items())
        },
        "species": species_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize accepted/rejected crop-quality outcomes."
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <manifest-dir>/quality_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.manifest_dir / "quality_report.json"
    report = build_quality_report(args.manifest_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
