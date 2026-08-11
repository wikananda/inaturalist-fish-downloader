"""CLI entrypoint for producing a broad fish species proposal."""

import argparse
import json
from pathlib import Path

from ..species.broad_plan import (
    build_species_proposal,
    load_plan_config,
    load_target_species,
    write_species_proposal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "broad_baseline_plan.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query iNaturalist counts and write a taxonomy-aware broad fish species "
            "proposal without downloading images."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target-species-file",
        type=Path,
        default=None,
        help="Override planning.target_species_file from the YAML plan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override planning.output_dir from the YAML plan.",
    )
    return parser.parse_args()


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_plan_config(_resolve_project_path(args.config))
    planning = config["planning"]
    target_path_value = args.target_species_file or planning.get("target_species_file")
    target_path = _resolve_project_path(target_path_value) if target_path_value else None
    output_dir = _resolve_project_path(args.output_dir or planning["output_dir"])

    target_species = load_target_species(target_path)
    rows, unmatched_targets = build_species_proposal(config, target_species)
    summary = write_species_proposal(output_dir, rows, unmatched_targets, planning)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote proposal to {output_dir / 'broad_species_proposal.tsv'}")
    if not summary["selected_download_species"]:
        raise SystemExit(
            "No species met the configured selection threshold. "
            f"The largest candidate global photo pool contained "
            f"{summary['max_global_observations']} observations, while "
            f"min_global_observations is {summary['min_global_observations']}. "
            "Calibrate the threshold against the filtered count inventory and the "
            "observed downloader acceptance rate before downloading."
        )
    minimum_train_species = int(planning.get("min_broad_train_species") or 0)
    if summary["broad_train_species"] < minimum_train_species:
        raise SystemExit(
            f"Plan selected {summary['broad_train_species']} broad training species, "
            f"below the configured minimum of {minimum_train_species}. Review the "
            "licensed count inventory, family scope, and selection caps."
        )


if __name__ == "__main__":
    main()
