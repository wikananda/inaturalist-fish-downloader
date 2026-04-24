"""iNaturalist-specific taxon and observation photo helpers."""

import argparse
import re
from typing import Iterator, Optional

from .config import ALIVE_OR_DEAD_TERM_ID, ALIVE_TERM_VALUE_ID
from .net import api_get


def resolve_taxon_id(
    species_name: str, include_subspecies: bool, retries: int = 5
) -> tuple[int, str]:
    """Resolve a species name to an iNaturalist taxon ID and canonical name."""
    payload = api_get("/taxa/autocomplete", q=species_name, per_page=30, retries=retries)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No taxon found for '{species_name}'")

    normalized = species_name.casefold()
    exact = []
    partial = []

    for item in results:
        candidate_names = {
            str(item.get("name", "")).casefold(),
            str(item.get("matched_term", "")).casefold(),
            str(item.get("preferred_common_name", "")).casefold(),
        }
        if normalized in candidate_names:
            exact.append(item)
        elif normalized in str(item.get("name", "")).casefold():
            partial.append(item)

    chosen = None
    if exact:
        chosen = exact[0]
    elif include_subspecies and partial:
        chosen = partial[0]
    elif include_subspecies and results:
        chosen = results[0]

    if chosen is None:
        raise ValueError(
            f"No exact taxon match for '{species_name}'. Try --include-subspecies if needed."
        )

    return int(chosen["id"]), str(chosen.get("name") or species_name)


def iter_observation_photos(
    taxon_id: int,
    quality_grade: str,
    per_page: int,
    max_pages: int,
    license_code: Optional[str],
    place_id: Optional[int],
    exclude_captive: bool,
    term_id: Optional[int],
    term_value_id: Optional[str],
    retries: int = 5,
    start_page: int = 1,
) -> Iterator[dict]:
    """Yield photo metadata from matching iNaturalist observations."""
    end_page = start_page + max_pages - 1
    for page in range(start_page, end_page + 1):
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "page": page,
            "per_page": per_page,
            "order_by": "votes",
            "order": "desc",
        }
        if quality_grade != "any":
            params["quality_grade"] = quality_grade
        if license_code:
            params["photo_license"] = license_code
        if place_id is not None:
            params["place_id"] = place_id
        if exclude_captive:
            params["captive"] = "false"
        if term_id is not None:
            params["term_id"] = term_id
        if term_value_id:
            params["term_value_id"] = term_value_id

        payload = api_get("/observations", retries=retries, **params)
        results = payload.get("results", [])
        if not results:
            return

        for observation in results:
            user = observation.get("user") or {}
            for photo in observation.get("photos", []):
                yield {
                    "_page": page,
                    "observation_id": observation.get("id"),
                    "photo_id": photo.get("id"),
                    "url": photo.get("url"),
                    "license_code": photo.get("license_code"),
                    "quality_grade": observation.get("quality_grade"),
                    "observed_on": observation.get("observed_on"),
                    "time_observed_at": observation.get("time_observed_at"),
                    "captive": observation.get("captive"),
                    "place_guess": observation.get("place_guess"),
                    "user_id": user.get("id"),
                    "user_login": user.get("login"),
                }


def photo_url_for_size(url: str, size: str) -> str:
    """Rewrite an iNaturalist photo URL to request a target size variant."""
    pattern = r"/(square|thumb|small|medium|large|original)\.(jpg|jpeg|png)$"
    return re.sub(pattern, rf"/{size}.\2", url, flags=re.IGNORECASE)


def infer_extension(url: str) -> str:
    """Infer an image file extension from a URL."""
    match = re.search(r"\.(jpg|jpeg|png)(?:\?|$)", url, flags=re.IGNORECASE)
    return f".{match.group(1).lower()}" if match else ".jpg"


def effective_annotation_filter(args: argparse.Namespace) -> tuple[Optional[int], Optional[str]]:
    """Resolve CLI annotation settings into API term filters."""
    if args.alive_only:
        return ALIVE_OR_DEAD_TERM_ID, str(ALIVE_TERM_VALUE_ID)
    return args.term_id, args.term_value_id
