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

from ..download.candidates import collect_photo_jobs, download_photo_job
from ..download.cli import output_paths, parse_args, validate_args
from ..download.detection import run_fish_detection
from ..download.image_quality import save_accepted_image, validate_image
from ..common.inat import resolve_taxon_id
from ..common.manifest import append_jsonl, append_species_summary
from ..common.utils import load_species, safe_print, slugify


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

    jobs = collect_photo_jobs(
        taxon_id=taxon_id,
        species_name=species_name,
        canonical_name=canonical_name,
        args=args,
        retries=args.retries,
    )
    if not jobs:
        safe_print("  no photos found")
        return

    candidates_path = manifest_dir / "candidates.jsonl"
    accepted_path = manifest_dir / "accepted.jsonl"
    rejected_path = manifest_dir / "rejected.jsonl"
    summary_path = manifest_dir / "species_summary.tsv"

    append_jsonl(
        candidates_path,
        [
            {
                **candidate,
                "raw_path": str(raw_species_dir / candidate["filename"]),
                "accepted_path": str(accepted_species_dir / candidate["filename"]),
            }
            for candidate in jobs
        ],
    )

    downloaded_by_photo_id = {}
    download_failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as executor:
        future_to_photo = {}
        for candidate in jobs:
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
                    candidate for candidate in jobs if candidate["photo_id"] == photo_id
                )
                failed_record = {
                    **failed,
                    "status": "failed",
                    "download_status": "failed",
                    "download_error": str(exc),
                    "raw_path": str(raw_species_dir / failed["filename"]),
                    "accepted_path": None,
                    "reject_reason": "download_failed",
                }
                download_failures.append(failed_record)
                safe_print(f"  failed photo {photo_id}: {exc}")

    accepted_records = []
    rejected_records = []
    unused_valid = 0

    for candidate in jobs:
        downloaded = downloaded_by_photo_id.get(candidate["photo_id"])
        if downloaded is None:
            continue

        raw_path = Path(downloaded["raw_path"])
        accepted_image_path = accepted_species_dir / candidate["filename"]
        record = {
            **downloaded,
            "accepted_path": str(accepted_image_path),
            "validation": {},
            "detection": {},
        }

        if args.skip_image_validation:
            is_valid, reject_reason, metrics = True, None, {}
        else:
            is_valid, reject_reason, metrics = validate_image(raw_path, args)

        record["validation"] = metrics

        if not is_valid:
            record["status"] = "rejected"
            record["reject_reason"] = reject_reason
            rejected_records.append(record)
            safe_print(f"  rejected: {candidate['filename']} ({reject_reason})")
            continue

        if len(accepted_records) >= args.images_per_species:
            record["status"] = "unused"
            record["reject_reason"] = "accepted_target_reached"
            rejected_records.append(record)
            unused_valid += 1
            continue

        if args.enable_detection:
            is_detected, reject_reason, detection_metrics = run_fish_detection(
                raw_path=raw_path,
                accepted_path=accepted_image_path,
                args=args,
            )
            record["detection"] = detection_metrics
            if not is_detected:
                record["status"] = "rejected"
                record["reject_reason"] = reject_reason
                rejected_records.append(record)
                safe_print(f"  rejected: {candidate['filename']} ({reject_reason})")
                continue
            accept_status = "accepted_crop"
        else:
            accept_status = save_accepted_image(
                raw_path=raw_path,
                accepted_path=accepted_image_path,
                overwrite=args.overwrite,
            )

        record["status"] = accept_status
        record["reject_reason"] = None
        accepted_records.append(record)
        safe_print(f"  {accept_status}: {accepted_image_path.name}")

    append_jsonl(accepted_path, accepted_records)
    append_jsonl(rejected_path, [*download_failures, *rejected_records])
    append_species_summary(
        summary_path,
        {
            "run_id": args.run_id,
            "species_name": species_name,
            "canonical_name": canonical_name,
            "taxon_id": taxon_id,
            "candidates": len(jobs),
            "downloaded": len(downloaded_by_photo_id),
            "download_failed": len(download_failures),
            "accepted": len(accepted_records),
            "rejected": len(download_failures) + len(rejected_records) - unused_valid,
            "unused_valid": unused_valid,
        },
    )

    safe_print(
        f"  accepted: {len(accepted_records)}/{args.images_per_species}; "
        f"candidates: {len(jobs)}; rejected: {len(rejected_records) - unused_valid}; "
        f"unused valid: {unused_valid}; "
        f"failed: {len(download_failures)}"
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
