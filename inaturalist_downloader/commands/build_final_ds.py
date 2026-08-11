"""CLI for building a final dataset from complete-enough species folders."""

import argparse
from pathlib import Path

from ..dataset.final_builder import build_final_dataset, write_records_tsv


def parse_args() -> argparse.Namespace:
    """Parse final dataset builder CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy species folders meeting a minimum image count into a final dataset."
        )
    )
    parser.add_argument(
        "threshold",
        type=int,
        help="Minimum number of images required for a species to be included.",
    )
    parser.add_argument(
        "--images-dir",
        default="filtered_ds",
        help="Source directory containing one folder per species. Default: filtered_ds",
    )
    parser.add_argument(
        "--output-dir",
        default="final_ds",
        help="Destination directory for included species folders. Default: final_ds",
    )
    parser.add_argument(
        "--included-report",
        default="manifests/final_ds_included.tsv",
        help="TSV report for species meeting threshold. Default: manifests/final_ds_included.tsv",
    )
    parser.add_argument(
        "--excluded-report",
        default="manifests/final_ds_excluded.tsv",
        help="TSV report for species below threshold. Default: manifests/final_ds_excluded.tsv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing species folders in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write reports and print actions without copying files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the final dataset builder command."""
    args = parse_args()
    try:
        result = build_final_dataset(
            Path(args.images_dir),
            Path(args.output_dir),
            args.threshold,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    write_records_tsv(result.included, Path(args.included_report))
    write_records_tsv(result.excluded, Path(args.excluded_report))

    mode = "Dry run" if args.dry_run else "Build"
    print(
        f"{mode}: {len(result.included)} species meet threshold; "
        f"{len(result.excluded)} below threshold."
    )
    print(
        f"Copied {result.copied_count} species folders; "
        f"skipped {result.skipped_existing_count} existing folders."
    )

    if result.included:
        print("\nSpecies meeting threshold:")
        for record in result.included:
            print(
                f"{record.species_name}: "
                f"{record.image_count}/{record.threshold} ({record.status})"
            )

    if result.excluded:
        print("\nSpecies below threshold:")
        for record in result.excluded:
            print(
                f"{record.species_name}: "
                f"{record.image_count}/{record.threshold} ({record.status})"
            )

    print(f"\nWrote included report to {args.included_report}")
    print(f"Wrote excluded report to {args.excluded_report}")


if __name__ == "__main__":
    main()
