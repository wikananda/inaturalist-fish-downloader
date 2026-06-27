"""CLI orchestration for downloading and filtering iNaturalist fish images.

The heavy responsibilities are split into focused modules:

- `cli`: argument parsing and validation
- `inat`: iNaturalist taxon/observation API helpers
- `candidates`: candidate photo collection and raw downloads
- `image_quality`: phase-2 image validation and accepted image writing
- `detection`: optional phase-3 YOLO detection and crop writing
- `manifest`: JSONL/TSV audit output
"""

import concurrent.futures
import time
from pathlib import Path

from ..download.candidates import (
    candidate_pages_per_batch,
    collect_photo_jobs,
    download_photo_job,
    remaining_candidate_capacity,
)
from ..download.cli import output_paths, parse_args, validate_args
from ..download.clip_filter import run_clip_filter
from ..download.detection import run_fish_detection_outputs
from ..download.image_quality import save_accepted_image, validate_image
from ..common.inat import resolve_taxon_id
from ..common.manifest import append_jsonl, append_species_summary
from ..common.utils import load_species, safe_print, slugify


def _planned_output_path(accepted_species_dir: Path, filename: str) -> str:
    """Return the deterministic output path for a candidate image."""
    return str(accepted_species_dir / filename)


def _update_output_state(
    record: dict,
    target_output_path: Path,
    *,
    saved_output: bool,
) -> None:
    """Attach final output-path state to a manifest record."""
    record["target_output_path"] = str(target_output_path)
    record["saved_output_path"] = str(target_output_path) if saved_output else None
    record["output_path_exists"] = target_output_path.exists()


def _license_search_plan(args) -> list[str | None]:
    """Return the ordered photo-license search plan for one species."""
    if args.license_preference:
        return list(args.license_preference)
    return [args.license_code]


def _is_blocked_license(record: dict, args) -> bool:
    license_code = record.get("license_code")
    return bool(
        license_code
        and str(license_code).strip().lower() in args.blocked_license_code_set
    )


def _reject_message(reason: str | None, metrics: dict | None = None) -> str:
    """Return a compact reject reason with the underlying tool error when present."""
    if metrics and metrics.get("error"):
        return f"{reason}: {metrics['error']}"
    return str(reason)


def download_species_images(
    species_name: str,
    args,
    output_dir: Path,
    raw_dir: Path,
    manifest_dir: Path,
) -> None:
    """Download, validate, and manifest images for one species."""
    taxon_id, canonical_name = resolve_taxon_id(
        species_name, include_subspecies=args.include_subspecies, retries=args.retries
    )
    species_slug = slugify(canonical_name)
    accepted_species_dir = output_dir / species_slug
    raw_species_dir = raw_dir / species_slug
    accepted_species_dir.mkdir(parents=True, exist_ok=True)
    raw_species_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"\n[{species_name}] taxon_id={taxon_id} -> {canonical_name}")

    candidates_path = manifest_dir / "candidates.jsonl"
    accepted_path = manifest_dir / "accepted.jsonl"
    rejected_path = manifest_dir / "rejected.jsonl"
    summary_path = manifest_dir / "species_summary.tsv"
    accepted_records = []
    rejected_records = []
    download_failures = []
    unused_valid = 0
    total_candidates_scanned = 0
    license_plan = _license_search_plan(args)
    next_pages = [1 for _ in license_plan]
    exhausted_licenses = [False for _ in license_plan]
    license_index = 0
    seen_photo_ids: set[int] = set()
    search_exhausted = False
    batch_index = 0

    while len(accepted_records) < args.images_per_species:
        while license_index < len(license_plan) and exhausted_licenses[license_index]:
            license_index += 1
        if license_index >= len(license_plan):
            search_exhausted = True
            break

        remaining_capacity = remaining_candidate_capacity(args, total_candidates_scanned)
        if remaining_capacity is not None and remaining_capacity <= 0:
            break

        current_license = license_plan[license_index]
        pages_to_scan = candidate_pages_per_batch(args)
        batch_index += 1
        batch_start_page = next_pages[license_index]
        jobs, next_page, batch_exhausted = collect_photo_jobs(
            taxon_id=taxon_id,
            species_name=species_name,
            canonical_name=canonical_name,
            args=args,
            start_page=batch_start_page,
            seen_photo_ids=seen_photo_ids,
            pages_to_scan=pages_to_scan,
            candidate_limit=remaining_capacity,
            retries=args.retries,
            license_code=current_license,
            license_priority=license_index + 1 if current_license else None,
        )
        next_pages[license_index] = next_page

        if not jobs:
            exhausted_licenses[license_index] = batch_exhausted or next_page > args.max_pages
            search_exhausted = all(exhausted_licenses)
            if search_exhausted and total_candidates_scanned == 0:
                safe_print("  no photos found")
            if exhausted_licenses[license_index]:
                license_index += 1
            if search_exhausted:
                break
            continue

        total_candidates_scanned += len(jobs)
        exhausted_licenses[license_index] = batch_exhausted or next_page > args.max_pages
        search_exhausted = all(exhausted_licenses)
        license_label = current_license or "any license"
        safe_print(
            f"  batch {batch_index}: {license_label} page {batch_start_page} "
            f"-> collected {len(jobs)} candidates"
        )

        append_jsonl(
            candidates_path,
            [
                {
                    **candidate,
                    "raw_path": str(raw_species_dir / candidate["filename"]),
                    "target_output_path": _planned_output_path(
                        accepted_species_dir, candidate["filename"]
                    ),
                    "saved_output_path": None,
                    "output_path_exists": (accepted_species_dir / candidate["filename"]).exists(),
                }
                for candidate in jobs
            ],
        )

        blocked_jobs = []
        downloadable_jobs = []
        for candidate in jobs:
            if _is_blocked_license(candidate, args):
                blocked_record = {
                    **candidate,
                    "status": "rejected",
                    "download_status": "skipped",
                    "download_error": None,
                    "raw_path": str(raw_species_dir / candidate["filename"]),
                    "reject_reason": "blocked_license",
                    "validation": {},
                    "detection": {},
                    "clip": {},
                }
                _update_output_state(
                    blocked_record,
                    accepted_species_dir / candidate["filename"],
                    saved_output=False,
                )
                blocked_jobs.append(blocked_record)
                safe_print(
                    f"  rejected: {candidate['filename']} "
                    f"(blocked_license: {candidate.get('license_code')})"
                )
            else:
                downloadable_jobs.append(candidate)
        rejected_records.extend(blocked_jobs)

        downloaded_by_photo_id = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.download_workers
        ) as executor:
            future_to_photo = {}
            for candidate in downloadable_jobs:
                photo_id = candidate["photo_id"]
                destination = raw_species_dir / candidate["filename"]
                future = executor.submit(
                    download_photo_job,
                    candidate,
                    destination,
                    args.overwrite,
                    args.sleep_seconds,
                    args.retries,
                )
                future_to_photo[future] = photo_id

            for future in concurrent.futures.as_completed(future_to_photo):
                photo_id = future_to_photo[future]
                try:
                    result = future.result()
                    downloaded_by_photo_id[photo_id] = result
                    safe_print(f"  {result['download_status']}: {result['filename']}")
                except Exception as exc:
                    failed = next(
                        candidate
                        for candidate in downloadable_jobs
                        if candidate["photo_id"] == photo_id
                    )
                    failed_record = {
                        **failed,
                        "status": "failed",
                        "download_status": "failed",
                        "download_error": str(exc),
                        "raw_path": str(raw_species_dir / failed["filename"]),
                        "reject_reason": "download_failed",
                    }
                    _update_output_state(
                        failed_record,
                        accepted_species_dir / failed["filename"],
                        saved_output=False,
                    )
                    download_failures.append(failed_record)
                    safe_print(f"  failed photo {photo_id}: {exc}")

        for candidate in downloadable_jobs:
            downloaded = downloaded_by_photo_id.get(candidate["photo_id"])
            if downloaded is None:
                continue

            raw_path = Path(downloaded["raw_path"])
            accepted_image_path = accepted_species_dir / candidate["filename"]
            record = {
                **downloaded,
                "validation": {},
                "detection": {},
                "clip": {},
            }
            _update_output_state(record, accepted_image_path, saved_output=False)

            if args.skip_image_validation:
                is_valid, reject_reason, metrics = True, None, {}
            else:
                is_valid, reject_reason, metrics = validate_image(raw_path, args)

            record["validation"] = metrics

            if not is_valid:
                record["status"] = "rejected"
                record["reject_reason"] = reject_reason
                _update_output_state(record, accepted_image_path, saved_output=False)
                rejected_records.append(record)
                safe_print(
                    f"  rejected: {candidate['filename']} "
                    f"({_reject_message(reject_reason, metrics)})"
                )
                continue

            if len(accepted_records) >= args.images_per_species:
                record["status"] = "unused"
                record["reject_reason"] = "accepted_target_reached"
                _update_output_state(record, accepted_image_path, saved_output=False)
                rejected_records.append(record)
                unused_valid += 1
                continue

            if args.enable_detection:
                remaining_slots = args.images_per_species - len(accepted_records)
                detection_outputs, reject_reason, detection_metrics = run_fish_detection_outputs(
                    raw_path=raw_path,
                    accepted_path=accepted_image_path,
                    args=args,
                    max_outputs=remaining_slots,
                )
                record["detection"] = detection_metrics
                if not detection_outputs:
                    record["status"] = "rejected"
                    record["reject_reason"] = reject_reason
                    _update_output_state(record, accepted_image_path, saved_output=False)
                    rejected_records.append(record)
                    safe_print(
                        f"  rejected: {candidate['filename']} "
                        f"({_reject_message(reject_reason, detection_metrics)})"
                    )
                    continue

                for detection_output in detection_outputs:
                    if len(accepted_records) >= args.images_per_species:
                        break

                    output_record = {
                        **record,
                        "filename": detection_output.accepted_path.name,
                        "detection": detection_output.metrics,
                        "clip": {},
                        "instance_index": detection_output.instance_index,
                        "instance_count": detection_output.instance_count,
                        "species_verification": detection_output.species_verification,
                    }
                    _update_output_state(
                        output_record,
                        detection_output.accepted_path,
                        saved_output=False,
                    )

                    if args.enable_clip_filter:
                        is_clip_ok, reject_reason, clip_metrics = run_clip_filter(
                            image_path=detection_output.clip_source_path,
                            args=args,
                        )
                        output_record["clip"] = clip_metrics
                        if not is_clip_ok:
                            if (
                                detection_output.created_output
                                and detection_output.accepted_path.exists()
                            ):
                                detection_output.accepted_path.unlink()
                            output_record["status"] = "rejected"
                            output_record["reject_reason"] = reject_reason
                            _update_output_state(
                                output_record,
                                detection_output.accepted_path,
                                saved_output=False,
                            )
                            rejected_records.append(output_record)
                            safe_print(
                                f"  rejected: {detection_output.accepted_path.name} "
                                f"({_reject_message(reject_reason, clip_metrics)})"
                            )
                            continue

                    output_record["status"] = detection_output.status
                    output_record["reject_reason"] = None
                    _update_output_state(
                        output_record,
                        detection_output.accepted_path,
                        saved_output=True,
                    )
                    accepted_records.append(output_record)
                    safe_print(f"  {detection_output.status}: {detection_output.accepted_path.name}")

                continue
            else:
                clip_source_path = raw_path

            if args.enable_clip_filter:
                is_clip_ok, reject_reason, clip_metrics = run_clip_filter(
                    image_path=clip_source_path,
                    args=args,
                )
                record["clip"] = clip_metrics
                if not is_clip_ok:
                    created_output = bool(record["detection"].get("created_output"))
                    if args.enable_detection and created_output and accepted_image_path.exists():
                        accepted_image_path.unlink()
                    record["status"] = "rejected"
                    record["reject_reason"] = reject_reason
                    _update_output_state(record, accepted_image_path, saved_output=False)
                    rejected_records.append(record)
                    safe_print(
                        f"  rejected: {candidate['filename']} "
                        f"({_reject_message(reject_reason, clip_metrics)})"
                    )
                    continue

            if not args.enable_detection:
                accept_status = save_accepted_image(
                    raw_path=raw_path,
                    accepted_path=accepted_image_path,
                    overwrite=args.overwrite,
                )

            record["status"] = accept_status
            record["reject_reason"] = None
            _update_output_state(record, accepted_image_path, saved_output=True)
            accepted_records.append(record)
            safe_print(f"  {accept_status}: {accepted_image_path.name}")

        if exhausted_licenses[license_index]:
            license_index += 1

    append_jsonl(accepted_path, accepted_records)
    append_jsonl(rejected_path, [*download_failures, *rejected_records])
    append_species_summary(
        summary_path,
        {
            "run_id": args.run_id,
            "species_name": species_name,
            "canonical_name": canonical_name,
            "taxon_id": taxon_id,
            "candidates": total_candidates_scanned,
            "scanned_candidates": total_candidates_scanned,
            "downloaded": total_candidates_scanned - len(download_failures),
            "download_failed": len(download_failures),
            "accepted": len(accepted_records),
            "rejected": len(download_failures) + len(rejected_records) - unused_valid,
            "unused_valid": unused_valid,
            "search_exhausted": search_exhausted,
        },
    )

    safe_print(
        f"  accepted: {len(accepted_records)}/{args.images_per_species}; "
        f"candidates: {total_candidates_scanned}; "
        f"rejected: {len(rejected_records) - unused_valid}; "
        f"unused valid: {unused_valid}; "
        f"failed: {len(download_failures)}; "
        f"search_exhausted: {search_exhausted}"
    )


def main() -> None:
    """Run the full downloader CLI workflow."""
    args = parse_args()
    validate_args(args)

    species_file, output_dir, raw_dir, manifest_dir = output_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.run_id = time.strftime("%Y%m%d-%H%M%S")

    if args.redownload:
        species_file = Path(args.redownload)
        args.overwrite = True
        safe_print(f"Redownload mode active: using {species_file} and forcing overwrite.")

    species_list = load_species(species_file)
    if not species_list:
        raise SystemExit(f"No species found in {species_file}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.species_workers
    ) as executor:
        future_to_species = {
            executor.submit(
                download_species_images,
                species_name,
                args,
                output_dir,
                raw_dir,
                manifest_dir,
            ): species_name
            for species_name in species_list
        }
        for future in concurrent.futures.as_completed(future_to_species):
            species_name = future_to_species[future]
            try:
                future.result()
            except Exception as exc:
                safe_print(f"\n[{species_name}] failed: {exc}")


if __name__ == "__main__":
    main()
