"""Core species-count extraction logic."""

import argparse
import time
from typing import Optional

from .api import http_get_json, resolve_taxon


def fetch_species_counts(args: argparse.Namespace, place_id: Optional[int]) -> list[dict]:
    """Fetch and filter species-count rows for the current taxon and place."""
    species_rows = []
    seen_taxon_ids = set()

    for page in range(1, args.max_pages + 1):
        params = {
            "taxon_id": args.taxon_id,
            "page": page,
            "per_page": args.per_page,
        }
        if place_id is not None:
            params["place_id"] = place_id
        if args.quality_grade != "any":
            params["quality_grade"] = args.quality_grade
        if args.photos_only:
            params["photos"] = "true"

        payload = http_get_json("/observations/species_counts", params)
        results = payload.get("results", [])
        if not results:
            break

        for row in results:
            taxon = row.get("taxon") or {}
            taxon_id = taxon.get("id")
            if not taxon_id or taxon_id in seen_taxon_ids:
                continue

            rank = str(taxon.get("rank") or "")
            if not args.include_lower_ranks and rank != "species":
                continue

            count = int(row.get("count") or 0)
            if count < args.min_observations:
                continue

            seen_taxon_ids.add(taxon_id)
            species_rows.append(
                {
                    "taxon_id": int(taxon_id),
                    "name": str(taxon.get("name") or "").strip(),
                    "rank": rank,
                    "count": count,
                    "preferred_common_name": str(
                        taxon.get("preferred_common_name") or ""
                    ).strip(),
                }
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    species_rows.sort(key=lambda item: (-item["count"], item["name"]))
    return species_rows


def get_species_for_place(
    args: argparse.Namespace,
    place_id: Optional[int],
    place_name: str,
    families: list[str],
) -> list[dict]:
    """Collect species rows for all configured families in one place scope."""
    print(f"[INFO] Searching in: {place_name} ({place_id})")
    all_species_rows = []

    for family_name in families:
        print(f"[INFO] Processing family: {family_name}...")
        try:
            family_id, resolved_family_name = resolve_taxon(family_name)
            print(f"      Resolved to: {resolved_family_name} ({family_id})")

            original_taxon_id = args.taxon_id
            args.taxon_id = family_id
            family_species = fetch_species_counts(args, place_id=place_id)
            args.taxon_id = original_taxon_id

            if args.species_per_family:
                family_species = family_species[: args.species_per_family]

            print(f"      Found {len(family_species)} species")
            all_species_rows.extend(family_species)

        except ValueError as e:
            print(f"      [WARN] Could not resolve family '{family_name}': {e}")
            continue

    return all_species_rows
