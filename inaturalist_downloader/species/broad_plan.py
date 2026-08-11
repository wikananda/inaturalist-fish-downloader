"""Taxonomy-aware planning for a broad regional fish baseline dataset."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

from .api import http_get_json


PROPOSAL_FIELDS = [
    "taxon_id",
    "species",
    "genus",
    "scientist_family",
    "inat_family",
    "preferred_common_name",
    "regional_count",
    "global_count",
    "eligible",
    "target_species",
    "selected",
    "dataset_role",
    "selection_reason",
    "estimated_accepted_observations",
    "planned_accepted_target",
]

TRAIN_ROLES = {"common_target_pretraining", "broad_pretraining"}
DOWNLOAD_ROLES = TRAIN_ROLES | {"novel_evaluation"}


def _assign_planned_targets(
    rows: list[dict[str, Any]], planning: dict[str, Any]
) -> None:
    """Attach pilot-calibrated yield estimates and progressive target tiers."""
    acceptance_rate = float(planning.get("expected_acceptance_rate") or 0.0)
    tiers = sorted(
        list(planning.get("accepted_target_tiers") or []),
        key=lambda tier: int(tier["min_global_observations"]),
    )
    novel_target = int(planning.get("novel_evaluation_accepted_target") or 0)

    for row in rows:
        global_count = int(row.get("global_count") or 0)
        row["estimated_accepted_observations"] = (
            round(global_count * acceptance_rate) if acceptance_rate > 0 else ""
        )
        row["planned_accepted_target"] = 0

        if row["dataset_role"] in TRAIN_ROLES:
            for tier in tiers:
                if global_count >= int(tier["min_global_observations"]):
                    row["planned_accepted_target"] = int(
                        tier["target_accepted_observations"]
                    )
        elif row["dataset_role"] == "novel_evaluation":
            row["planned_accepted_target"] = novel_target


def load_plan_config(path: Path) -> dict[str, Any]:
    """Load and validate the broad-baseline YAML plan."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Broad-baseline plan not found: {config_path}")
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Broad-baseline plan must contain a mapping")
    planning = payload.get("planning")
    families = payload.get("families")
    if not isinstance(planning, dict) or not isinstance(families, list) or not families:
        raise ValueError("Plan requires non-empty 'planning' and 'families' sections")
    return payload


def load_target_species(path: Path | None) -> set[str]:
    """Load requested species names using case-insensitive matching."""
    if path is None:
        return set()
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target species file not found: {target_path}")
    return {
        line.strip().casefold()
        for line in target_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def fetch_species_counts(
    *,
    taxon_id: int,
    place_id: int | None,
    quality_grade: str,
    photos_only: bool,
    per_page: int,
    max_pages: int,
    retries: int,
    sleep_seconds: float,
    photo_license_codes: list[str] | None = None,
    exclude_captive: bool = False,
) -> dict[int, dict[str, Any]]:
    """Return species-count rows keyed by taxon ID for one search scope."""
    rows: dict[int, dict[str, Any]] = {}
    for page in range(1, max_pages + 1):
        params: dict[str, Any] = {
            "taxon_id": taxon_id,
            "page": page,
            "per_page": per_page,
        }
        if place_id is not None:
            params["place_id"] = place_id
        if quality_grade != "any":
            params["quality_grade"] = quality_grade
        if photos_only:
            params["photos"] = "true"
        if photo_license_codes:
            params["photo_license"] = ",".join(photo_license_codes)
        if exclude_captive:
            params["captive"] = "false"

        payload = http_get_json(
            "/observations/species_counts",
            params,
            retries=retries,
        )
        results = payload.get("results", [])
        if not results:
            break
        for result in results:
            taxon = result.get("taxon") or {}
            if taxon.get("rank") != "species" or not taxon.get("id"):
                continue
            current_id = int(taxon["id"])
            rows[current_id] = {
                "taxon_id": current_id,
                "species": str(taxon.get("name") or "").strip(),
                "preferred_common_name": str(
                    taxon.get("preferred_common_name") or ""
                ).strip(),
                "count": int(result.get("count") or 0),
            }
        if len(results) < per_page:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def _genus_for_species(species: str) -> str:
    parts = species.split()
    return parts[0] if parts else ""


def _collect_candidate_rows(
    config: dict[str, Any],
    fetcher=fetch_species_counts,
) -> list[dict[str, Any]]:
    planning = config["planning"]
    combined: dict[tuple[str, int], dict[str, Any]] = {}

    for family in config["families"]:
        scientist_family = str(family["scientist_family"])
        for inat_taxon in family.get("inat_taxa", []):
            inat_family = str(inat_taxon["name"])
            family_taxon_id = int(inat_taxon["taxon_id"])
            shared_filters = {
                "quality_grade": str(planning["quality_grade"]),
                "photos_only": bool(planning["photos_only"]),
                "per_page": int(planning["per_page"]),
                "max_pages": int(planning["max_pages"]),
                "retries": int(planning["retries"]),
                "sleep_seconds": float(planning["sleep_seconds"]),
                "photo_license_codes": list(
                    planning.get("photo_license_codes") or []
                ),
                "exclude_captive": bool(planning.get("exclude_captive", False)),
            }
            if planning.get("collect_regional_counts", True):
                regional = fetcher(
                    taxon_id=family_taxon_id,
                    place_id=int(planning["region_place_id"]),
                    **shared_filters,
                )
            else:
                regional = {}
            global_rows = fetcher(
                taxon_id=family_taxon_id,
                place_id=None,
                **shared_filters,
            )
            for taxon_id in set(regional) | set(global_rows):
                regional_row = regional.get(taxon_id, {})
                global_row = global_rows.get(taxon_id, {})
                species = str(
                    regional_row.get("species") or global_row.get("species") or ""
                ).strip()
                if not species:
                    continue
                key = (scientist_family, taxon_id)
                candidate = combined.setdefault(
                    key,
                    {
                        "taxon_id": taxon_id,
                        "species": species,
                        "genus": _genus_for_species(species),
                        "scientist_family": scientist_family,
                        "inat_family": inat_family,
                        "preferred_common_name": str(
                            regional_row.get("preferred_common_name")
                            or global_row.get("preferred_common_name")
                            or ""
                        ),
                        "regional_count": 0,
                        "global_count": 0,
                    },
                )
                candidate["regional_count"] = max(
                    int(candidate["regional_count"]), int(regional_row.get("count") or 0)
                )
                candidate["global_count"] = max(
                    int(candidate["global_count"]), int(global_row.get("count") or 0)
                )
    return list(combined.values())


def select_species(
    rows: list[dict[str, Any]],
    planning: dict[str, Any],
    target_species: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Apply abundance, family, genus, target, and novel-species rules."""
    min_regional = int(planning["min_regional_observations"])
    min_global = int(planning["min_global_observations"])
    family_limit = int(planning["max_species_per_scientist_family"])
    genus_limit = int(planning["max_species_per_genus"])
    novel_fraction = float(planning["novel_evaluation_fraction"])
    require_regional = bool(planning.get("require_regional_threshold", True))
    rng = random.Random(int(planning["random_seed"]))

    found_targets = {row["species"].casefold() for row in rows} & target_species
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["target_species"] = row["species"].casefold() in target_species
        row["eligible"] = (
            int(row["global_count"]) >= min_global
            and (
                not require_regional
                or int(row["regional_count"]) >= min_regional
            )
        )
        row["selected"] = False
        row["dataset_role"] = (
            "eligible_not_selected" if row["eligible"] else "ineligible"
        )
        row["selection_reason"] = (
            "below_required_observation_threshold"
            if not row["eligible"]
            else "eligible_not_selected"
        )
        grouped[row["scientist_family"]].append(row)

    for family_rows in grouped.values():
        eligible = [row for row in family_rows if row["eligible"]]
        eligible.sort(
            key=lambda row: (
                not row["target_species"],
                -int(row["global_count"]),
                -int(row["regional_count"]),
                row["species"],
            )
        )

        selected: list[dict[str, Any]] = []
        genus_counts: Counter[str] = Counter()

        # Eligible requested species are always retained. Family/genus caps control
        # only the extra broad-pretraining species added around them.
        for row in eligible:
            if not row["target_species"]:
                continue
            row["selected"] = True
            row["dataset_role"] = "common_target_pretraining"
            row["selection_reason"] = "requested_target_meets_licensed_abundance_threshold"
            selected.append(row)
            genus_counts[row["genus"]] += 1

        for row in eligible:
            if row["selected"]:
                continue
            if len(selected) >= family_limit:
                row["selection_reason"] = "scientist_family_quota_reached"
                continue
            if genus_counts[row["genus"]] >= genus_limit:
                row["selection_reason"] = "genus_quota_reached"
                continue
            row["selected"] = True
            row["dataset_role"] = "broad_pretraining"
            row["selection_reason"] = "global_licensed_abundance"
            selected.append(row)
            genus_counts[row["genus"]] += 1

        rare_targets = [
            row for row in family_rows if row["target_species"] and not row["eligible"]
        ]
        for row in rare_targets:
            row["dataset_role"] = "rare_target_holdout"
            row["selection_reason"] = "requested_target_below_abundance_thresholds"

        novel_pool = [
            row
            for row in selected
            if row["dataset_role"] == "broad_pretraining"
        ]
        if len(novel_pool) >= 4 and novel_fraction > 0:
            novel_count = max(1, round(len(novel_pool) * novel_fraction))
            rng.shuffle(novel_pool)
            for row in novel_pool[:novel_count]:
                row["dataset_role"] = "novel_evaluation"
                row["selection_reason"] = "held_out_for_unseen_species_evaluation"

    _assign_planned_targets(rows, planning)

    role_order = {
        "common_target_pretraining": 0,
        "broad_pretraining": 1,
        "novel_evaluation": 2,
        "rare_target_holdout": 3,
        "eligible_not_selected": 4,
        "ineligible": 5,
    }
    rows.sort(
        key=lambda row: (
            row["scientist_family"],
            role_order.get(row["dataset_role"], 99),
            -int(row["regional_count"]),
            row["species"],
        )
    )
    return rows, target_species - found_targets


def build_species_proposal(
    config: dict[str, Any],
    target_species: set[str],
    fetcher=fetch_species_counts,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Fetch count inventories and build the complete proposal."""
    rows = _collect_candidate_rows(config, fetcher=fetcher)
    return select_species(rows, config["planning"], target_species)


def _write_name_list(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    names = sorted({str(row["species"]) for row in rows})
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def write_species_proposal(
    output_dir: Path,
    rows: list[dict[str, Any]],
    unmatched_targets: set[str],
    planning: dict[str, Any],
) -> dict[str, Any]:
    """Write the proposal TSV, role lists, unmatched targets, and summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    proposal_path = output / "broad_species_proposal.tsv"
    with proposal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PROPOSAL_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PROPOSAL_FIELDS})

    train_rows = [row for row in rows if row["dataset_role"] in TRAIN_ROLES]
    novel_rows = [row for row in rows if row["dataset_role"] == "novel_evaluation"]
    rare_rows = [row for row in rows if row["dataset_role"] == "rare_target_holdout"]
    download_rows = [row for row in rows if row["dataset_role"] in DOWNLOAD_ROLES]
    _write_name_list(output / "broad_train_species.txt", train_rows)
    _write_name_list(output / "novel_evaluation_species.txt", novel_rows)
    _write_name_list(output / "rare_target_species.txt", rare_rows)
    _write_name_list(output / "selected_download_species.txt", download_rows)
    target_tiers = sorted(
        {
            int(tier["target_accepted_observations"])
            for tier in planning.get("accepted_target_tiers") or []
        }
    )
    for target in target_tiers:
        tier_rows = [
            row
            for row in train_rows
            if int(row.get("planned_accepted_target") or 0) >= target
        ]
        _write_name_list(output / f"target_{target}_train_species.txt", tier_rows)
    (output / "unmatched_target_species.txt").write_text(
        "\n".join(sorted(unmatched_targets)) + ("\n" if unmatched_targets else ""),
        encoding="utf-8",
    )

    role_counts = Counter(row["dataset_role"] for row in rows)
    family_counts = Counter(
        row["scientist_family"] for row in rows if row["dataset_role"] in DOWNLOAD_ROLES
    )
    eligible_count = sum(bool(row.get("eligible")) for row in rows)
    max_global_count = max(
        (int(row.get("global_count") or 0) for row in rows),
        default=0,
    )
    target_counts = Counter(
        int(row.get("planned_accepted_target") or 0) for row in train_rows
    )
    target_counts.pop(0, None)
    planned_total = sum(
        int(row.get("planned_accepted_target") or 0) for row in train_rows
    )
    projected_retrievable = sum(
        min(
            int(row.get("planned_accepted_target") or 0),
            int(row.get("estimated_accepted_observations") or 0),
        )
        for row in train_rows
    )
    summary = {
        "plan_name": planning["name"],
        "region_name": planning["region_name"],
        "region_place_id": planning["region_place_id"],
        "min_regional_observations": planning["min_regional_observations"],
        "min_global_observations": planning["min_global_observations"],
        "require_regional_threshold": planning.get(
            "require_regional_threshold", True
        ),
        "photo_license_codes": list(planning.get("photo_license_codes") or []),
        "candidate_species": len(rows),
        "eligible_species": eligible_count,
        "max_global_observations": max_global_count,
        "expected_acceptance_rate": planning.get("expected_acceptance_rate"),
        "selected_download_species": len(download_rows),
        "broad_train_species": len(train_rows),
        "novel_evaluation_species": len(novel_rows),
        "rare_target_species": len(rare_rows),
        "unmatched_target_species": len(unmatched_targets),
        "role_counts": dict(sorted(role_counts.items())),
        "selected_by_scientist_family": dict(sorted(family_counts.items())),
        "planned_train_by_accepted_target": {
            str(target): count for target, count in sorted(target_counts.items())
        },
        "planned_train_accepted_observations": planned_total,
        "projected_retrievable_at_planned_targets": projected_retrievable,
        "count_note": (
            "Observation counts already apply the configured research-grade, photo, "
            "licence, and captive filters; accepted counts will still be lower after "
            "image, detector, semantic, diversity, and deduplication gates."
        ),
    }
    (output / "plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
