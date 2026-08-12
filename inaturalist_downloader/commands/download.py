"""Download, filter, and persist accepted-target iNaturalist fish datasets."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..common.inat import resolve_taxon_id
from ..common.manifest import append_jsonl, append_species_summary
from ..common.utils import SpeciesRequest, load_species_requests, safe_print, slugify
from ..download.candidates import (
    adaptive_candidate_batch_limit,
    candidate_pages_per_batch,
    collect_photo_jobs,
    download_photo_job,
    remaining_candidate_capacity,
)
from ..download.cli import (
    effective_config_yaml,
    output_paths,
    parse_args,
    validate_args,
)
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
from ..download.dedup import DatasetDeduplicator
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
        "license_validation": {
            "enforce_allowed": getattr(args, "enforce_allowed_licenses", False),
            "allowed": sorted(_allowed_license_codes(args)),
            "blocked": sorted(getattr(args, "blocked_license_code_set", set())),
        },
        "require_taxon_membership": getattr(args, "require_taxon_membership", False),
        "require_exact_species_taxon": getattr(
            args, "require_exact_species_taxon", False
        ),
        "dataset_dedup": {
            field: getattr(args, field, None)
            for field in (
                "enable_dataset_dedup",
                "deduplicate_observation_ids",
                "deduplicate_photo_ids",
                "deduplicate_exact_content",
                "deduplicate_perceptual_content",
                "perceptual_hash_distance",
            )
        },
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
            "device": getattr(args, "detector_device", None),
            "confidence": getattr(args, "detector_confidence", None),
            "imgsz": getattr(args, "detector_imgsz", None),
            "class_names": sorted(
                getattr(args, "detector_class_name_set", set()) or set()
            ),
            "class_ids": sorted(
                getattr(args, "detector_class_id_set", set()) or set()
            ),
            "min_fish_area_ratio": getattr(args, "min_fish_area_ratio", None),
            "crop_padding": getattr(args, "crop_padding", None),
            "allow_multiple_fish": getattr(args, "allow_multiple_fish", None),
            "sam_score_threshold": getattr(args, "sam_score_threshold", None),
            "sam_prompt": getattr(args, "sam_prompt", None),
            "sam_max_instances_per_image": getattr(
                args, "sam_max_instances_per_image", None
            ),
            "sam_min_mask_area_ratio": getattr(
                args, "sam_min_mask_area_ratio", None
            ),
            "sam_min_yolo_iou": getattr(args, "sam_min_yolo_iou", None),
            "sam_allow_yolo_fallback": getattr(
                args, "sam_allow_yolo_fallback", None
            ),
            "sam_crop_padding": getattr(args, "sam_crop_padding", None),
            "sam_save_all_instances": getattr(args, "sam_save_all_instances", None),
            "sam_checkpoint_path": getattr(args, "sam_checkpoint_path", None),
            "sam_dtype": getattr(args, "sam_dtype", None),
            "sam_autocast": getattr(args, "sam_autocast", None),
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
        and str(license_code).strip().lower()
        in getattr(args, "blocked_license_code_set", set())
    )


def _allowed_license_codes(args) -> set[str]:
    configured = getattr(args, "allowed_license_code_set", None)
    if configured is not None:
        return {
            str(value).strip().lower()
            for value in configured
            if str(value).strip()
        }
    values = set(getattr(args, "license_preference", []) or [])
    license_code = getattr(args, "license_code", None)
    if license_code:
        values.add(license_code)
    return {
        str(value).strip().lower() for value in values if str(value).strip()
    }


def _license_reject_reason(record: dict[str, Any], args) -> str | None:
    if _is_blocked_license(record, args):
        return "blocked_license"
    if not getattr(args, "enforce_allowed_licenses", False):
        return None
    license_code = str(record.get("license_code") or "").strip().lower()
    if not license_code:
        return "missing_photo_license"
    if license_code not in _allowed_license_codes(args):
        return "disallowed_photo_license"
    return None


def _taxon_reject_reason(record: dict[str, Any], args) -> str | None:
    require_exact = getattr(args, "require_exact_species_taxon", False)
    if not require_exact and not getattr(args, "require_taxon_membership", False):
        return None
    requested_taxon_id = record.get("requested_taxon_id") or record.get("taxon_id")
    observation_taxon_id = record.get("observation_taxon_id")
    if observation_taxon_id is None:
        return "missing_observation_taxon"
    requested = int(requested_taxon_id)
    observed = int(observation_taxon_id)
    ancestor_ids = {
        int(value)
        for value in (record.get("observation_ancestor_ids") or [])
        if value is not None
    }
    if require_exact and observed != requested:
        infraspecific_ranks = {
            "subspecies",
            "variety",
            "form",
        }
        observed_rank = str(record.get("observation_taxon_rank") or "").casefold()
        if requested not in ancestor_ids or observed_rank not in infraspecific_ranks:
            return "observation_species_taxon_mismatch"
    if observed != requested and requested not in ancestor_ids:
        return "observation_taxon_mismatch"
    return None


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


def _bootstrap_progress_from_manifest(
    progress,
    accepted_path: Path,
    *,
    taxon_id: int,
    canonical_name: str,
) -> int:
    """Seed a new exact-taxon state from already accepted manifest records."""
    if not accepted_path.exists() or progress.target_count("observation") > 0:
        return 0
    seeded = 0
    with accepted_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_taxon_id = record.get("training_taxon_id") or record.get("taxon_id")
            if int(record_taxon_id or 0) != taxon_id:
                continue
            if str(record.get("canonical_name") or "") != canonical_name:
                continue
            output_path = record.get("saved_output_path")
            if not output_path or not Path(output_path).exists():
                continue
            observation_id = record.get("observation_id")
            photo_id = record.get("photo_id")
            if observation_id is None or photo_id is None:
                continue
            if int(observation_id) in progress.accepted_observation_ids:
                continue
            progress.seen_observation_ids.add(int(observation_id))
            progress.seen_photo_ids.add(int(photo_id))
            progress.mark_accepted(record)
            seeded += 1
    return seeded


def download_species_images(
    species_name: str,
    args,
    output_dir: Path,
    raw_dir: Path,
    manifest_dir: Path,
    dataset_deduplicator: DatasetDeduplicator | None = None,
    *,
    requested_taxon_id: int | None = None,
    accepted_target: int | None = None,
) -> None:
    """Refill one species until its accepted image/observation target is met.

    A structured plan passes ``requested_taxon_id`` and bypasses autocomplete.
    ``accepted_target`` overrides the profile's fallback target for this class.
    """
    if requested_taxon_id is None:
        taxon_id, canonical_name = resolve_taxon_id(
            species_name,
            include_subspecies=args.include_subspecies,
            retries=args.retries,
        )
    else:
        taxon_id, canonical_name = int(requested_taxon_id), species_name
    target = int(accepted_target or args.images_per_species)
    if args.max_candidates_per_species is not None and target > args.max_candidates_per_species:
        raise ValueError(
            f"Accepted target {target} for {species_name} exceeds "
            f"max_candidates_per_species={args.max_candidates_per_species}"
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
    if dataset_deduplicator is None:
        dataset_deduplicator = DatasetDeduplicator(
            accepted_path,
            enabled=getattr(args, "enable_dataset_dedup", False),
            observation_ids=getattr(args, "deduplicate_observation_ids", True),
            photo_ids=getattr(args, "deduplicate_photo_ids", True),
            exact_content=getattr(args, "deduplicate_exact_content", True),
            perceptual_content=getattr(
                args, "deduplicate_perceptual_content", True
            ),
            perceptual_distance=getattr(args, "perceptual_hash_distance", 0),
        )

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
    if bool(getattr(args, "resume", False) and not args.overwrite):
        seeded = _bootstrap_progress_from_manifest(
            progress,
            accepted_path,
            taxon_id=taxon_id,
            canonical_name=canonical_name,
        )
        if seeded:
            safe_print(f"  bootstrapped {seeded} accepted observations from manifest")
            save_progress(state_path, progress)
    if (
        getattr(args, "refresh_exhausted_scopes", False)
        and progress.target_count(target_unit) < target
        and all(progress.exhausted_scopes.values())
    ):
        progress.refresh_exhausted_scopes()
        safe_print(
            "  refreshing exhausted scopes from page 1; accepted and seen IDs "
            "are preserved"
        )
    stop_reason = None

    while progress.target_count(target_unit) < target:
        active_scope = next(
            (scope for scope in scopes if not progress.exhausted_scopes[scope["key"]]),
            None,
        )
        if active_scope is None:
            stop_reason = "search_space_exhausted"
            break

        remaining_capacity = remaining_candidate_capacity(
            args, progress.candidate_budget_scanned
        )
        if remaining_capacity is not None and remaining_capacity <= 0:
            stop_reason = "candidate_budget_exhausted"
            break

        accepted_before = progress.target_count(target_unit)
        remaining_target = target - accepted_before
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
        progress.candidate_budget_scanned += len(jobs)
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
            reject_reason = _license_reject_reason(candidate, args)
            if reject_reason is None:
                reject_reason = _taxon_reject_reason(candidate, args)
            dedup_decision = None
            if reject_reason is None:
                dedup_decision = dataset_deduplicator.check_source_identity(candidate)
                if not dedup_decision.accepted:
                    reject_reason = dedup_decision.reason
            if reject_reason is None:
                downloadable_jobs.append(candidate)
                continue
            record = {
                **candidate,
                "status": "rejected",
                "download_status": "skipped",
                "download_error": None,
                "raw_path": str(raw_species_dir / candidate["filename"]),
                "reject_reason": reject_reason,
                "validation": {},
                "detection": {},
                "clip": {},
                "dedup": dedup_decision.metrics if dedup_decision else {},
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

            if progress.target_count(target_unit) >= target:
                record["status"] = "unused"
                record["reject_reason"] = "accepted_target_reached"
                batch_rejected.append(record)
                group["records"].append(record)
                progress.unused_valid += 1
                continue

            if args.enable_detection:
                max_outputs = getattr(args, "max_crops_per_observation", None)
                if max_outputs is None and target_unit == "image":
                    max_outputs = target - progress.target_count(target_unit)
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

            if progress.target_count(target_unit) >= target:
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

            dedup_decision = dataset_deduplicator.check_and_register(
                record,
                Path(pending["output_path"]),
            )
            record["dedup"] = dedup_decision.metrics
            if not dedup_decision.accepted:
                # This path belongs to the incoming candidate. Remove it even if
                # it was left by an interrupted pre-manifest run; otherwise a
                # rejected duplicate could still leak into folder-based training.
                duplicate_output_path = Path(pending["output_path"])
                if duplicate_output_path.exists():
                    duplicate_output_path.unlink()
                record["status"] = "rejected"
                record["reject_reason"] = dedup_decision.reason
                _update_output_state(
                    record, Path(pending["output_path"]), saved_output=False
                )
                batch_rejected.append(record)
                group["records"].append(record)
                progress.rejected += 1
                continue

            record["content_sha256"] = dedup_decision.metrics.get("content_sha256")
            record["perceptual_dhash"] = dedup_decision.metrics.get(
                "perceptual_dhash"
            )

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
            f"{progress.target_count(target_unit)}/{target}; "
            f"candidates scanned: {progress.candidates_scanned}"
        )

    if progress.target_count(target_unit) >= target:
        stop_reason = "target_reached"
    elif stop_reason is None:
        stop_reason = (
            "search_space_exhausted"
            if all(progress.exhausted_scopes.values())
            else "stopped"
        )
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
            "target": target,
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
        f"{target}; outputs: {progress.accepted_outputs}; "
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
    (manifest_dir / "effective_config.yaml").write_text(
        effective_config_yaml(args.effective_config), encoding="utf-8"
    )
    if args.redownload:
        species_file = Path(args.redownload)
        args.overwrite = True
        args.resume = False
        safe_print(f"Redownload mode active: using {species_file} and forcing overwrite.")

    append_jsonl(
        manifest_dir / "run_history.jsonl",
        [
            {
                "run_id": args.run_id,
                "config_path": args.config_path,
                "species_file": str(species_file),
                "output_dir": str(output_dir),
                "raw_dir": str(raw_dir),
                "manifest_dir": str(manifest_dir),
                "resume": bool(args.resume and not args.overwrite),
                "redownload": bool(args.redownload),
            }
        ],
    )

    species_requests = load_species_requests(species_file)
    if not species_requests:
        raise SystemExit(f"No species found in {species_file}")
    exact_request_count = sum(request.taxon_id is not None for request in species_requests)
    per_species_target_count = sum(request.target is not None for request in species_requests)
    safe_print(
        f"Loaded {len(species_requests)} species requests: "
        f"{exact_request_count} exact taxon IDs, "
        f"{per_species_target_count} per-species targets."
    )
    if getattr(args, "require_exact_species_taxon", False) and (
        exact_request_count != len(species_requests)
    ):
        raise SystemExit(
            "This profile requires exact species taxa, but the species file has "
            f"{len(species_requests) - exact_request_count} row(s) without taxon_id. "
            "Use a TSV/CSV plan with taxon_id, species, and optional target columns."
        )
    invalid_targets = [
        request
        for request in species_requests
        if request.target is not None
        and args.max_candidates_per_species is not None
        and request.target > args.max_candidates_per_species
    ]
    if invalid_targets:
        raise SystemExit(
            "Species-plan targets cannot exceed max_candidates_per_species: "
            + ", ".join(
                f"{request.species}={request.target}" for request in invalid_targets[:10]
            )
        )

    dataset_deduplicator = DatasetDeduplicator(
        manifest_dir / "accepted.jsonl",
        enabled=getattr(args, "enable_dataset_dedup", False),
        observation_ids=getattr(args, "deduplicate_observation_ids", True),
        photo_ids=getattr(args, "deduplicate_photo_ids", True),
        exact_content=getattr(args, "deduplicate_exact_content", True),
        perceptual_content=getattr(args, "deduplicate_perceptual_content", True),
        perceptual_distance=getattr(args, "perceptual_hash_distance", 0),
    )

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
        failures = []
        future_to_request = {
            executor.submit(
                download_species_images,
                request.species,
                args,
                output_dir,
                raw_dir,
                manifest_dir,
                dataset_deduplicator,
                requested_taxon_id=request.taxon_id,
                accepted_target=request.target,
            ): request
            for request in species_requests
        }
        for future in concurrent.futures.as_completed(future_to_request):
            request = future_to_request[future]
            species_name = request.species
            try:
                future.result()
            except Exception as exc:
                safe_print(f"\n[{species_name}] failed: {exc}")
                failures.append(
                    {
                        "run_id": args.run_id,
                        "species_name": species_name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    if failures:
        append_jsonl(manifest_dir / "failures.jsonl", failures)
        raise SystemExit(
            f"{len(failures)} species failed. Details: "
            f"{manifest_dir / 'failures.jsonl'}. Re-run the same command to resume."
        )


if __name__ == "__main__":
    main()
