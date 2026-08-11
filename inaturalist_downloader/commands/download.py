"""Download, filter, and persist accepted-target iNaturalist fish datasets."""

from __future__ import annotations

import concurrent.futures
import hashlib
import time
from pathlib import Path
from typing import Any

from ..common.inat import resolve_taxon_id
from ..common.manifest import append_jsonl, append_species_summary
from ..common.utils import load_species, safe_print, slugify
from ..download.candidates import (
    adaptive_candidate_batch_limit,
    candidate_pages_per_batch,
    collect_photo_jobs,
    download_photo_job,
    remaining_candidate_capacity,
)
from ..download.cli import output_paths, parse_args, validate_args
from ..download.clip_filter import (
    preload_clip_model,
    run_clip_filter,
    run_clip_filter_batch,
)
from ..download.detection import (
    get_detector_model,
    preload_sam3_model,
    run_fish_detection_outputs,
)
from ..download.image_quality import save_accepted_image, validate_image
from ..download.progress import (
    load_progress,
    save_progress,
    search_scope_key,
)


def _planned_output_path(accepted_species_dir: Path, filename: str) -> str:
    return str(accepted_species_dir / filename)


def _update_output_state(
    record: dict[str, Any],
    target_output_path: Path,
    *,
    saved_output: bool,
) -> None:
    record["target_output_path"] = str(target_output_path)
    record["saved_output_path"] = str(target_output_path) if saved_output else None
    record["output_path_exists"] = target_output_path.exists()


def _license_search_plan(args) -> list[str | None]:
    if args.license_preference:
        return list(args.license_preference)
    return [args.license_code]


def _observation_search_plan(args) -> list[dict[str, Any]]:
    places = list(getattr(args, "place_preference", []) or [args.place_id])
    scopes = []
    seen_keys = set()
    for place_id in places:
        for priority, license_code in enumerate(_license_search_plan(args), start=1):
            key = search_scope_key(place_id, license_code)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            scopes.append(
                {
                    "key": key,
                    "place_id": place_id,
                    "license_code": license_code,
                    "license_priority": priority if license_code else None,
                }
            )
    return scopes


def _search_signature(taxon_id: int, args, scopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe every setting that changes candidate identity or acceptance."""
    return {
        "taxon_id": taxon_id,
        "scopes": [
            {"place_id": scope["place_id"], "license_code": scope["license_code"]}
            for scope in scopes
        ],
        "quality_grade": args.quality_grade,
        "photo_size": args.photo_size,
        "exclude_captive": args.exclude_captive,
        "term_id": args.term_id,
        "term_value_id": args.term_value_id,
        "order_by": args.order_by,
        "order": args.order,
        "query_params": args.query_params,
        "target_unit": getattr(args, "target_unit", "image"),
        "max_photos_per_observation": getattr(args, "max_photos_per_observation", None),
        "max_crops_per_observation": getattr(args, "max_crops_per_observation", None),
        "max_images_per_observer_per_species": getattr(
            args, "max_images_per_observer_per_species", None
        ),
        "validation": {
            "skip": args.skip_image_validation,
            "min_width": getattr(args, "min_width", None),
            "min_height": getattr(args, "min_height", None),
            "min_file_size_kb": getattr(args, "min_file_size_kb", None),
            "max_aspect_ratio": getattr(args, "max_aspect_ratio", None),
            "min_intensity_range": getattr(args, "min_intensity_range", None),
        },
        "detection": {
            "enabled": args.enable_detection,
            "backend": getattr(args, "detection_backend", None),
            "weights": getattr(args, "detector_weights", None),
            "confidence": getattr(args, "detector_confidence", None),
            "crop_padding": getattr(args, "crop_padding", None),
            "allow_multiple_fish": getattr(args, "allow_multiple_fish", None),
            "sam_score_threshold": getattr(args, "sam_score_threshold", None),
            "sam_min_mask_area_ratio": getattr(
                args, "sam_min_mask_area_ratio", None
            ),
            "sam_min_yolo_iou": getattr(args, "sam_min_yolo_iou", None),
            "sam_allow_yolo_fallback": getattr(
                args, "sam_allow_yolo_fallback", None
            ),
        },
        "crop_quality": {
            field: getattr(args, field, None)
            for field in (
                "enable_crop_quality",
                "crop_min_short_side",
                "crop_min_long_side",
                "min_fish_bbox_width",
                "min_fish_bbox_height",
                "min_fish_crop_area_ratio",
                "max_fish_crop_area_ratio",
                "min_source_edge_margin_ratio",
                "min_crop_edge_margin_ratio",
                "min_crop_edge_variance",
                "min_crop_entropy",
                "sam_min_mask_crop_area_ratio",
                "sam_max_mask_crop_area_ratio",
                "crop_redetect",
                "crop_redetect_confidence",
                "crop_redetect_require_single",
                "crop_redetect_min_area_ratio",
                "crop_redetect_max_area_ratio",
                "crop_redetect_min_iou",
                "crop_redetect_min_edge_margin_ratio",
            )
        },
        "clip": {
            "enabled": args.enable_clip_filter,
            "backend": getattr(args, "clip_backend", "clip"),
            "model": getattr(args, "clip_model", None),
            "threshold": getattr(args, "clip_threshold", None),
            "prompts_file": getattr(args, "clip_prompts_file", None),
        },
    }


def _is_blocked_license(record: dict[str, Any], args) -> bool:
    license_code = record.get("license_code")
    return bool(
        license_code
        and str(license_code).strip().lower() in args.blocked_license_code_set
    )


def _should_keep_rejected_raw(record: dict[str, Any], args) -> bool:
    policy = getattr(args, "rejected_raw_policy", "keep")
    if policy == "keep":
        return True
    if policy == "delete":
        return False
    stable_key = "|".join(
        str(record.get(field) or "")
        for field in ("taxon_id", "observation_id", "photo_id")
    )
    digest = hashlib.sha256(stable_key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return fraction < float(getattr(args, "rejected_raw_sample_rate", 0.05))


def _apply_rejected_raw_policy(
    records: list[dict[str, Any]],
    raw_path: Path,
    args,
    *,
    source_accepted: bool,
) -> None:
    keep = source_accepted or _should_keep_rejected_raw(records[0], args)
    if not keep and raw_path.exists():
        raw_path.unlink()
    retained = raw_path.exists()
    for record in records:
        record["raw_retained"] = retained


def _observer_cap_reached(progress, record: dict[str, Any], args) -> bool:
    cap = getattr(args, "max_images_per_observer_per_species", None)
    user_id = record.get("user_id")
    if cap is None or user_id is None:
        return False
    return progress.accepted_by_user.get(str(user_id), 0) >= cap


def _observation_crop_cap_reached(progress, record: dict[str, Any], args) -> bool:
    cap = getattr(args, "max_crops_per_observation", None)
    observation_id = record.get("observation_id")
    if cap is None or observation_id is None:
        return False
    return progress.accepted_crops_by_observation.get(str(observation_id), 0) >= cap


def _remove_created_output(pending: dict[str, Any]) -> None:
    output_path = pending.get("output_path")
    if pending.get("created_output") and output_path and Path(output_path).exists():
        Path(output_path).unlink()


def _clip_pending_outputs(pending: list[dict[str, Any]], args):
    if not args.enable_clip_filter:
        return [(True, None, {}) for _ in pending]
    paths = [Path(item["clip_path"]) for item in pending]
    batch_size = int(getattr(args, "clip_batch_size", 32))
    results = []
    for offset in range(0, len(paths), batch_size):
        chunk = paths[offset : offset + batch_size]
        # Preserve the single-image function as a simple/testable path while
        # using true batching for normal refill batches.
        if len(chunk) == 1:
            results.append(run_clip_filter(chunk[0], args))
        else:
            results.extend(run_clip_filter_batch(chunk, args))
    return results


def _flush_batch(
    accepted_path: Path,
    rejected_path: Path,
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> None:
    append_jsonl(accepted_path, accepted_records)
    append_jsonl(rejected_path, rejected_records)


def download_species_images(
    species_name: str,
    args,
    output_dir: Path,
    raw_dir: Path,
    manifest_dir: Path,
) -> None:
    """Refill one species until its accepted image/observation target is met."""
    taxon_id, canonical_name = resolve_taxon_id(
        species_name,
        include_subspecies=args.include_subspecies,
        retries=args.retries,
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
    state_path = manifest_dir / "state" / f"{species_slug}.json"

    scopes = _observation_search_plan(args)
    scope_keys = [scope["key"] for scope in scopes]
    signature = _search_signature(taxon_id, args, scopes)
    progress = load_progress(
        state_path,
        signature=signature,
        scope_keys=scope_keys,
        resume=bool(getattr(args, "resume", False) and not args.overwrite),
    )
    target_unit = getattr(args, "target_unit", "image")
    stop_reason = None

    while progress.target_count(target_unit) < args.images_per_species:
        active_scope = next(
            (scope for scope in scopes if not progress.exhausted_scopes[scope["key"]]),
            None,
        )
        if active_scope is None:
            stop_reason = "api_exhausted"
            break

        remaining_capacity = remaining_candidate_capacity(
            args, progress.candidates_scanned
        )
        if remaining_capacity is not None and remaining_capacity <= 0:
            stop_reason = "candidate_budget_exhausted"
            break

        accepted_before = progress.target_count(target_unit)
        remaining_target = args.images_per_species - accepted_before
        batch_limit = adaptive_candidate_batch_limit(
            args,
            remaining_target=remaining_target,
            accepted_count=accepted_before,
            processed_count=progress.candidates_scanned,
        )
        if remaining_capacity is not None:
            batch_limit = min(batch_limit, remaining_capacity)
        pages_to_scan = candidate_pages_per_batch(args, batch_limit)
        progress.batch_index += 1
        scope_key = active_scope["key"]
        batch_start_page = progress.next_pages[scope_key]

        jobs, next_page, batch_exhausted = collect_photo_jobs(
            taxon_id=taxon_id,
            species_name=species_name,
            canonical_name=canonical_name,
            args=args,
            start_page=batch_start_page,
            seen_photo_ids=progress.seen_photo_ids,
            seen_observation_ids=progress.seen_observation_ids,
            pages_to_scan=pages_to_scan,
            candidate_limit=batch_limit,
            retries=args.retries,
            license_code=active_scope["license_code"],
            license_priority=active_scope["license_priority"],
            place_id_override=active_scope["place_id"],
        )
        if remaining_capacity is not None and len(jobs) > remaining_capacity:
            jobs = jobs[:remaining_capacity]
        progress.next_pages[scope_key] = next_page
        progress.exhausted_scopes[scope_key] = (
            batch_exhausted or next_page > args.max_pages
        )

        if not jobs:
            save_progress(state_path, progress)
            continue

        progress.candidates_scanned += len(jobs)
        place_label = active_scope["place_id"] or "global"
        license_label = active_scope["license_code"] or "any license"
        safe_print(
            f"  batch {progress.batch_index}: place={place_label}, {license_label}, "
            f"page {batch_start_page} -> {len(jobs)} candidates"
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
                    "output_path_exists": (
                        accepted_species_dir / candidate["filename"]
                    ).exists(),
                }
                for candidate in jobs
            ],
        )

        batch_accepted: list[dict[str, Any]] = []
        batch_rejected: list[dict[str, Any]] = []
        downloadable_jobs = []
        for candidate in jobs:
            if not _is_blocked_license(candidate, args):
                downloadable_jobs.append(candidate)
                continue
            record = {
                **candidate,
                "status": "rejected",
                "download_status": "skipped",
                "download_error": None,
                "raw_path": str(raw_species_dir / candidate["filename"]),
                "reject_reason": "blocked_license",
                "validation": {},
                "detection": {},
                "clip": {},
                "raw_retained": False,
            }
            _update_output_state(
                record, accepted_species_dir / candidate["filename"], saved_output=False
            )
            batch_rejected.append(record)
            progress.rejected += 1

        downloaded_by_photo_id = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.download_workers
        ) as executor:
            future_to_candidate = {
                executor.submit(
                    download_photo_job,
                    candidate,
                    raw_species_dir / candidate["filename"],
                    args.overwrite,
                    args.sleep_seconds,
                    args.retries,
                ): candidate
                for candidate in downloadable_jobs
            }
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    result = future.result()
                    downloaded_by_photo_id[candidate["photo_id"]] = result
                    progress.downloaded += 1
                except Exception as exc:
                    failed_record = {
                        **candidate,
                        "status": "failed",
                        "download_status": "failed",
                        "download_error": str(exc),
                        "raw_path": str(raw_species_dir / candidate["filename"]),
                        "reject_reason": "download_failed",
                        "raw_retained": False,
                    }
                    _update_output_state(
                        failed_record,
                        accepted_species_dir / candidate["filename"],
                        saved_output=False,
                    )
                    batch_rejected.append(failed_record)
                    progress.download_failed += 1

        pending_outputs: list[dict[str, Any]] = []
        raw_groups: dict[int, dict[str, Any]] = {}
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
            group = raw_groups.setdefault(
                int(candidate["photo_id"]),
                {"raw_path": raw_path, "records": [], "accepted": False},
            )

            if args.skip_image_validation:
                is_valid, reject_reason, metrics = True, None, {}
            else:
                is_valid, reject_reason, metrics = validate_image(raw_path, args)
            record["validation"] = metrics
            if not is_valid:
                record["status"] = "rejected"
                record["reject_reason"] = reject_reason
                batch_rejected.append(record)
                group["records"].append(record)
                progress.rejected += 1
                continue

            if progress.target_count(target_unit) >= args.images_per_species:
                record["status"] = "unused"
                record["reject_reason"] = "accepted_target_reached"
                batch_rejected.append(record)
                group["records"].append(record)
                progress.unused_valid += 1
                continue

            if args.enable_detection:
                max_outputs = getattr(args, "max_crops_per_observation", None)
                if max_outputs is None and target_unit == "image":
                    max_outputs = args.images_per_species - progress.target_count(target_unit)
                detection_outputs, reject_reason, detection_metrics = (
                    run_fish_detection_outputs(
                        raw_path=raw_path,
                        accepted_path=accepted_image_path,
                        args=args,
                        max_outputs=max_outputs,
                    )
                )
                record["detection"] = detection_metrics
                if not detection_outputs:
                    record["status"] = "rejected"
                    record["reject_reason"] = reject_reason
                    batch_rejected.append(record)
                    group["records"].append(record)
                    progress.rejected += 1
                    continue

                for detection_output in detection_outputs:
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
                        output_record, detection_output.accepted_path, saved_output=False
                    )
                    pending_outputs.append(
                        {
                            "record": output_record,
                            "clip_path": detection_output.clip_source_path,
                            "output_path": detection_output.accepted_path,
                            "created_output": detection_output.created_output,
                            "accept_status": detection_output.status,
                            "needs_save": False,
                            "group": group,
                        }
                    )
            else:
                pending_outputs.append(
                    {
                        "record": record,
                        "clip_path": raw_path,
                        "output_path": accepted_image_path,
                        "created_output": False,
                        "accept_status": "accepted",
                        "needs_save": True,
                        "group": group,
                    }
                )

        clip_results = _clip_pending_outputs(pending_outputs, args)
        for pending, (clip_ok, reject_reason, clip_metrics) in zip(
            pending_outputs, clip_results
        ):
            record = pending["record"]
            group = pending["group"]
            record["clip"] = clip_metrics

            if not clip_ok:
                _remove_created_output(pending)
                record["status"] = "rejected"
                record["reject_reason"] = reject_reason
                _update_output_state(record, Path(pending["output_path"]), saved_output=False)
                batch_rejected.append(record)
                group["records"].append(record)
                progress.rejected += 1
                continue

            if progress.target_count(target_unit) >= args.images_per_species:
                _remove_created_output(pending)
                record["status"] = "unused"
                record["reject_reason"] = "accepted_target_reached"
                _update_output_state(record, Path(pending["output_path"]), saved_output=False)
                batch_rejected.append(record)
                group["records"].append(record)
                progress.unused_valid += 1
                continue

            if target_unit == "observation" and record.get("observation_id") in (
                progress.accepted_observation_ids
            ):
                _remove_created_output(pending)
                record["status"] = "unused"
                record["reject_reason"] = "duplicate_accepted_observation"
                _update_output_state(record, Path(pending["output_path"]), saved_output=False)
                batch_rejected.append(record)
                group["records"].append(record)
                progress.unused_valid += 1
                continue

            if _observation_crop_cap_reached(progress, record, args):
                _remove_created_output(pending)
                record["status"] = "unused"
                record["reject_reason"] = "observation_crop_cap_reached"
                _update_output_state(record, Path(pending["output_path"]), saved_output=False)
                batch_rejected.append(record)
                group["records"].append(record)
                progress.unused_valid += 1
                continue

            if _observer_cap_reached(progress, record, args):
                _remove_created_output(pending)
                record["status"] = "rejected"
                record["reject_reason"] = "observer_cap_reached"
                _update_output_state(record, Path(pending["output_path"]), saved_output=False)
                batch_rejected.append(record)
                group["records"].append(record)
                progress.rejected += 1
                continue

            if pending["needs_save"]:
                accept_status = save_accepted_image(
                    raw_path=Path(record["raw_path"]),
                    accepted_path=Path(pending["output_path"]),
                    overwrite=args.overwrite,
                )
            else:
                accept_status = pending["accept_status"]
            record["status"] = accept_status
            record["reject_reason"] = None
            _update_output_state(record, Path(pending["output_path"]), saved_output=True)
            batch_accepted.append(record)
            group["accepted"] = True
            progress.mark_accepted(record)

        for photo_id, group in raw_groups.items():
            records = group["records"]
            if records:
                _apply_rejected_raw_policy(
                    records,
                    group["raw_path"],
                    args,
                    source_accepted=group["accepted"],
                )
            for record in batch_accepted:
                if record.get("photo_id") is not None and int(record["photo_id"]) == photo_id:
                    record["raw_retained"] = group["raw_path"].exists()

        _flush_batch(accepted_path, rejected_path, batch_accepted, batch_rejected)
        save_progress(state_path, progress)
        safe_print(
            f"  batch accepted: {len(batch_accepted)}; target progress: "
            f"{progress.target_count(target_unit)}/{args.images_per_species}; "
            f"candidates scanned: {progress.candidates_scanned}"
        )

    if progress.target_count(target_unit) >= args.images_per_species:
        stop_reason = "target_reached"
    elif stop_reason is None:
        stop_reason = "api_exhausted" if all(progress.exhausted_scopes.values()) else "stopped"
    progress.stop_reason = stop_reason
    save_progress(state_path, progress)

    append_species_summary(
        summary_path,
        {
            "run_id": args.run_id,
            "species_name": species_name,
            "canonical_name": canonical_name,
            "taxon_id": taxon_id,
            "target_unit": target_unit,
            "target": args.images_per_species,
            "candidates": progress.candidates_scanned,
            "scanned_candidates": progress.candidates_scanned,
            "downloaded": progress.downloaded,
            "download_failed": progress.download_failed,
            "accepted": progress.target_count(target_unit),
            "accepted_outputs": progress.accepted_outputs,
            "accepted_observations": len(progress.accepted_observation_ids),
            "rejected": progress.rejected,
            "unused_valid": progress.unused_valid,
            "search_exhausted": all(progress.exhausted_scopes.values()),
            "stop_reason": stop_reason,
        },
    )
    safe_print(
        f"  accepted {target_unit}s: {progress.target_count(target_unit)}/"
        f"{args.images_per_species}; outputs: {progress.accepted_outputs}; "
        f"candidates: {progress.candidates_scanned}; stop_reason: {stop_reason}"
    )


def main() -> None:
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
        args.resume = False
        safe_print(f"Redownload mode active: using {species_file} and forcing overwrite.")

    species_list = load_species(species_file)
    if not species_list:
        raise SystemExit(f"No species found in {species_file}")

    if args.enable_detection:
        backend = args.detection_backend
        if backend in ("yolo", "cascade"):
            try:
                get_detector_model(args.detector_weights)
                safe_print(f"YOLO detector ready: {args.detector_weights}")
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        if backend in ("sam3", "cascade") and args.sam_preload:
            safe_print(
                f"Preparing SAM 3 model files in {args.sam_model_dir} "
                f"from {args.sam_repo_id}..."
            )
            try:
                checkpoint_path = preload_sam3_model(args)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            safe_print(f"SAM 3 ready: {checkpoint_path}")

    if args.enable_clip_filter:
        safe_print(f"Preparing CLIP model {args.clip_model}...")
        try:
            preload_clip_model(args)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        safe_print(f"CLIP ready: {args.clip_model}")

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
