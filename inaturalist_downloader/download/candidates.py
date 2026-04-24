"""Candidate collection and raw photo download jobs."""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

from ..common.inat import (
    effective_annotation_filter,
    infer_extension,
    iter_observation_photos,
    photo_url_for_size,
)
from ..common.net import download_file
from ..common.utils import slugify


def candidate_batch_limit_for_args(args: argparse.Namespace) -> int:
    """Calculate how many candidates to request in one refill batch."""
    candidate_limit = max(
        args.images_per_species,
        math.ceil(args.images_per_species * args.candidate_multiplier),
    )
    if args.max_candidates_per_species is not None:
        candidate_limit = min(candidate_limit, args.max_candidates_per_species)
    return candidate_limit


def candidate_pages_per_batch(args: argparse.Namespace) -> int:
    """Estimate how many observation pages to scan in one refill batch."""
    return max(1, math.ceil(candidate_batch_limit_for_args(args) / max(args.per_page, 1)))


def remaining_candidate_capacity(
    args: argparse.Namespace,
    scanned_candidates: int,
) -> Optional[int]:
    """Return remaining scan capacity for a species, or None if uncapped."""
    if args.max_candidates_per_species is None:
        return None
    return max(0, args.max_candidates_per_species - scanned_candidates)


def collect_photo_jobs(
    taxon_id: int,
    species_name: str,
    canonical_name: str,
    args: argparse.Namespace,
    start_page: int,
    seen_photo_ids: set[int],
    pages_to_scan: int,
    candidate_limit: Optional[int] = None,
    retries: int = 5,
) -> tuple[list[dict], int, bool]:
    """Build one refill batch of candidate photo records for a species."""
    jobs = []
    species_slug = slugify(canonical_name)
    term_id, term_value_id = effective_annotation_filter(args)
    remaining_pages = args.max_pages - start_page + 1
    if remaining_pages <= 0 or pages_to_scan <= 0:
        return jobs, args.max_pages + 1, True

    actual_pages_to_scan = min(pages_to_scan, remaining_pages)
    expected_end_page = start_page + actual_pages_to_scan - 1
    last_page = start_page - 1

    for photo in iter_observation_photos(
        taxon_id=taxon_id,
        quality_grade=args.quality_grade,
        per_page=args.per_page,
        max_pages=actual_pages_to_scan,
        license_code=args.license_code,
        place_id=args.place_id,
        exclude_captive=args.exclude_captive,
        term_id=term_id,
        term_value_id=term_value_id,
        retries=retries,
        start_page=start_page,
    ):
        last_page = int(photo.get("_page", last_page))
        photo_id = photo.get("photo_id")
        raw_url = photo.get("url")
        observation_id = photo.get("observation_id")
        if not photo_id or not raw_url or photo_id in seen_photo_ids:
            continue

        seen_photo_ids.add(photo_id)
        image_url = photo_url_for_size(raw_url, args.photo_size)
        filename = (
            f"{species_slug}__obs_{observation_id}__photo_{photo_id}"
            f"{infer_extension(image_url)}"
        )
        jobs.append(
            {
                "run_id": args.run_id,
                "species_name": species_name,
                "canonical_name": canonical_name,
                "taxon_id": taxon_id,
                "observation_id": observation_id,
                "photo_id": photo_id,
                "photo_url": image_url,
                "source_photo_url": raw_url,
                "filename": filename,
                "license_code": photo.get("license_code"),
                "quality_grade": photo.get("quality_grade"),
                "place_id": args.place_id,
                "observed_on": photo.get("observed_on"),
                "time_observed_at": photo.get("time_observed_at"),
                "captive": photo.get("captive"),
                "place_guess": photo.get("place_guess"),
                "user_id": photo.get("user_id"),
                "user_login": photo.get("user_login"),
                "status": "candidate",
                "reject_reason": None,
                "scores": {},
            }
        )
        if candidate_limit is not None and len(jobs) >= candidate_limit:
            break

    next_page = min(args.max_pages + 1, expected_end_page + 1)
    exhausted = last_page < expected_end_page or next_page > args.max_pages

    return jobs, next_page, exhausted


def download_photo_job(
    candidate: dict,
    destination: Path,
    overwrite: bool,
    sleep_seconds: float,
    retries: int = 5,
) -> dict:
    """Download one candidate photo and return an updated candidate record."""
    did_download = download_file(
        url=candidate["photo_url"],
        destination=destination,
        overwrite=overwrite,
        retries=retries,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    status = "downloaded" if did_download else "skipped"
    result = dict(candidate)
    result.update(
        {
            "raw_path": str(destination),
            "download_status": status,
            "download_error": None,
        }
    )
    return result
