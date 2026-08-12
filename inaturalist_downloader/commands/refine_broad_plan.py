"""Refine a broad species plan with observed downloader yields."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPOSAL = PROJECT_ROOT / "plans/broad_coral_global_v3/broad_species_proposal.tsv"
DEFAULT_SUMMARY = PROJECT_ROOT / "manifests/broad_coral_global_v3/species_summary.tsv"
DEFAULT_ACCEPTED = PROJECT_ROOT / "manifests/broad_coral_global_v3/accepted.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "plans/broad_coral_global_v4"
DEFAULT_CONTEXT_EXCLUSIONS = PROJECT_ROOT / "configs/broad_context_risk_species.txt"

ELIGIBLE_ROLES = {
    "common_target_pretraining",
    "broad_pretraining",
    "eligible_not_selected",
}

OUTPUT_FIELDS = [
    "taxon_id",
    "species",
    "genus",
    "scientist_family",
    "inat_family",
    "global_count",
    "v3_role",
    "v3_accepted",
    "exact_label_accepted",
    "metadata_mismatch_count",
    "family_smoothed_yield",
    "estimated_exact_accepted",
    "planned_accepted_target",
    "v4_role",
    "selection_reason",
]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_names(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _write_plan(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    sort_key=None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["taxon_id", "species", "target"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sorted(rows, key=sort_key or (lambda item: item["species"])):
            writer.writerow(
                {
                    "taxon_id": row["taxon_id"],
                    "species": row["species"],
                    "target": row["planned_accepted_target"],
                }
            )


def refine_plan(
    proposal_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    accepted_rows: list[dict[str, Any]],
    *,
    minimum_exact_accepted: int = 100,
    projected_minimum: float = 110.0,
    reserve_projected_minimum: float = 80.0,
    collection_target: int = 200,
    prior_yield: float = 0.288292,
    prior_candidates: int = 500,
    measured_reserve_minimum: int = 65,
    context_exclusions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Classify V3 candidates into primary, reserve, context-risk, or excluded."""
    context_exclusions = context_exclusions or set()
    proposal_by_species = {row["species"]: row for row in proposal_rows}
    v3_accepted = {
        row["canonical_name"]: int(row["accepted"])
        for row in summary_rows
        if row.get("canonical_name")
    }

    family_accepted: Counter[str] = Counter()
    family_scanned: Counter[str] = Counter()
    for row in summary_rows:
        proposal = proposal_by_species.get(row.get("canonical_name", ""))
        if not proposal:
            continue
        family = proposal["inat_family"]
        family_accepted[family] += int(row.get("accepted") or 0)
        family_scanned[family] += int(
            row.get("scanned_candidates") or row.get("candidates") or 0
        )

    proposal_by_taxon_id = {
        int(row["taxon_id"]): row for row in proposal_rows if row.get("taxon_id")
    }
    exact_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    for row in accepted_rows:
        training_label = str(row.get("canonical_name") or "")
        proposal = proposal_by_species.get(training_label)
        observed_taxon_id = row.get("observation_taxon_id")
        observed_taxon_id_int = (
            int(observed_taxon_id) if observed_taxon_id is not None else None
        )
        observed_proposal = (
            proposal_by_taxon_id.get(observed_taxon_id_int)
            if observed_taxon_id_int is not None
            else None
        )
        if observed_proposal is not None:
            exact_counts[observed_proposal["species"]] += 1
        if not proposal:
            continue
        planned_taxon_id = int(proposal["taxon_id"])
        if observed_taxon_id_int == planned_taxon_id:
            continue
        else:
            # Infraspecific observations are safe descendants of an exact species.
            rank = str(row.get("observation_taxon_rank") or "").casefold()
            ancestors = {int(value) for value in row.get("observation_ancestor_ids") or []}
            if planned_taxon_id in ancestors and rank in {
                "subspecies",
                "variety",
                "form",
            }:
                exact_counts[training_label] += 1
            else:
                mismatch_counts[training_label] += 1

    output: list[dict[str, Any]] = []
    for proposal in proposal_rows:
        species = proposal["species"]
        exact_seed = exact_counts.get(species, 0)
        if (
            proposal.get("dataset_role") not in ELIGIBLE_ROLES
            and exact_seed < measured_reserve_minimum
        ):
            continue
        family = proposal["inat_family"]
        global_count = int(proposal.get("global_count") or 0)
        denominator = family_scanned[family] + prior_candidates
        smoothed_yield = (
            (family_accepted[family] + prior_yield * prior_candidates) / denominator
            if denominator
            else prior_yield
        )
        estimated = global_count * smoothed_yield
        measured = v3_accepted.get(species)
        exact = exact_seed if exact_seed or measured is not None else None
        mismatch = mismatch_counts.get(species, 0)
        excluded_for_context = species.casefold() in context_exclusions

        if excluded_for_context:
            role = "context_risk_holdout"
            reason = "class_concentrated_dead_caught_held_or_specimen_context"
            target = 0
        elif exact is not None and exact >= minimum_exact_accepted:
            role = "primary_train"
            reason = "v3_exact_label_count_meets_minimum"
            target = collection_target
        elif measured is None and estimated >= projected_minimum:
            role = "primary_train"
            reason = (
                "relabelled_seed_plus_supply_projected_to_reach_minimum"
                if exact
                else "unselected_v3_candidate_projected_to_reach_minimum"
            )
            target = collection_target
        elif (
            (exact is not None and exact >= measured_reserve_minimum)
            or estimated >= reserve_projected_minimum
        ):
            role = "reserve_backfill"
            reason = "near_minimum_or_projection_uncertain"
            target = collection_target
        else:
            role = "excluded_low_supply"
            reason = "unlikely_to_reach_100_after_current_gates"
            target = 0

        output.append(
            {
                "taxon_id": int(proposal["taxon_id"]),
                "species": species,
                "genus": proposal["genus"],
                "scientist_family": proposal["scientist_family"],
                "inat_family": family,
                "global_count": global_count,
                "v3_role": proposal["dataset_role"],
                "v3_accepted": measured if measured is not None else "",
                "exact_label_accepted": exact if exact is not None else "",
                "metadata_mismatch_count": mismatch,
                "family_smoothed_yield": round(smoothed_yield, 6),
                "estimated_exact_accepted": round(estimated),
                "planned_accepted_target": target,
                "v4_role": role,
                "selection_reason": reason,
            }
        )
    return output


def write_refined_plan(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "broad_species_proposal.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in OUTPUT_FIELDS} for row in rows)

    primary = [row for row in rows if row["v4_role"] == "primary_train"]
    reserve = [row for row in rows if row["v4_role"] == "reserve_backfill"]
    context = [row for row in rows if row["v4_role"] == "context_risk_holdout"]
    _write_plan(output_dir / "broad_train_species.tsv", primary)
    _write_plan(
        output_dir / "reserve_backfill_species.tsv",
        reserve,
        sort_key=lambda row: (
            0 if isinstance(row["exact_label_accepted"], int) else 1,
            -int(row["exact_label_accepted"] or 0),
            -int(row["estimated_exact_accepted"] or 0),
            row["species"],
        ),
    )
    _write_plan(output_dir / "primary_and_reserve_species.tsv", primary + reserve)
    _write_plan(
        output_dir / "target_200_train_species.tsv",
        [row for row in primary if int(row["planned_accepted_target"]) >= 200],
    )
    (output_dir / "context_risk_holdout_species.txt").write_text(
        "\n".join(sorted(row["species"] for row in context)) + ("\n" if context else ""),
        encoding="utf-8",
    )

    role_counts = Counter(row["v4_role"] for row in rows)
    target_counts = Counter(
        int(row["planned_accepted_target"])
        for row in primary
        if int(row["planned_accepted_target"]) > 0
    )
    summary = {
        "plan_name": "broad_coral_fish_global_pretraining_v4_empirical",
        "primary_train_species": len(primary),
        "reserve_backfill_species": len(reserve),
        "context_risk_holdout_species": len(context),
        "role_counts": dict(sorted(role_counts.items())),
        "primary_by_target": {
            str(target): count for target, count in sorted(target_counts.items())
        },
        "primary_planned_accepted_observations": sum(
            int(row["planned_accepted_target"]) for row in primary
        ),
        "collection_target_per_selected_species": (
            int(primary[0]["planned_accepted_target"]) if primary else None
        ),
        "recommended_post_download_thresholds": [100, 150, 200],
        "primary_existing_exact_at_least_100": sum(
            isinstance(row["exact_label_accepted"], int)
            and row["exact_label_accepted"] >= 100
            for row in primary
        ),
        "new_primary_species": sum(row["v3_accepted"] == "" for row in primary),
        "method_note": (
            "Primary selection uses exact-label V3 counts where available and a "
            "family-smoothed empirical yield for previously unselected species. "
            "Reserve classes are downloaded only if the primary set finishes below "
            "the desired breadth; its TSV is ordered by exact V3 count and then "
            "projected supply. Projections are estimates, not guarantees."
        ),
    }
    (output_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a broad species plan using observed accepted-image yields."
    )
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--species-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--accepted-manifest", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-exclusions", type=Path, default=DEFAULT_CONTEXT_EXCLUSIONS)
    parser.add_argument("--minimum-exact-accepted", type=int, default=100)
    parser.add_argument("--projected-minimum", type=float, default=110.0)
    parser.add_argument("--reserve-projected-minimum", type=float, default=80.0)
    parser.add_argument(
        "--collection-target",
        type=int,
        default=200,
        help=(
            "Attempted accepted-observation ceiling for every selected class; "
            "assess completed folders at 100/150/200 afterward."
        ),
    )
    parser.add_argument("--prior-yield", type=float, default=0.288292)
    parser.add_argument("--prior-candidates", type=int, default=500)
    parser.add_argument("--measured-reserve-minimum", type=int, default=65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = refine_plan(
        _read_tsv(args.proposal),
        _read_tsv(args.species_summary),
        _read_jsonl(args.accepted_manifest),
        minimum_exact_accepted=args.minimum_exact_accepted,
        projected_minimum=args.projected_minimum,
        reserve_projected_minimum=args.reserve_projected_minimum,
        collection_target=args.collection_target,
        prior_yield=args.prior_yield,
        prior_candidates=args.prior_candidates,
        measured_reserve_minimum=args.measured_reserve_minimum,
        context_exclusions=_load_names(args.context_exclusions),
    )
    print(json.dumps(write_refined_plan(args.output_dir, rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
