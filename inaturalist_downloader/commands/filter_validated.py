"""CLI for extracting Label Studio validated images into a filtered dataset."""

import argparse
from pathlib import Path

from ..dataset.validated_filter import extract_validated_images, status_count


MATERIALIZED_STATUSES = {"copied", "symlinked", "hardlinked", "would_copy", "would_symlink", "would_hardlink"}


def parse_args() -> argparse.Namespace:
    """Parse validated-dataset filtering arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Label Studio validated images into class folders."
    )
    parser.add_argument(
        "--csv",
        default="dataset.csv",
        help="Label Studio CSV export path. Default: dataset.csv",
    )
    parser.add_argument(
        "--images-dir",
        default="downloads",
        help="Source image directory with class folders. Default: downloads",
    )
    parser.add_argument(
        "--output-dir",
        default="filtered_ds",
        help="Filtered output dataset directory. Default: filtered_ds",
    )
    parser.add_argument(
        "--report",
        default="manifests/validated_filter_report.jsonl",
        help="JSONL report path. Default: manifests/validated_filter_report.jsonl",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="CSV column containing Label Studio image references. Default: image",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="CSV column containing class labels. Default: label",
    )
    parser.add_argument(
        "--valid-column",
        default="valid",
        help="CSV column containing validation label. Default: valid",
    )
    parser.add_argument(
        "--valid-value",
        default="validated",
        help="Value treated as accepted/validated. Default: validated",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize validated images. Default: copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the report and summary without creating files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the validated image extractor."""
    args = parse_args()
    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    summary = extract_validated_images(
        csv_path=csv_path,
        images_dir=images_dir,
        output_dir=output_dir,
        report_path=report_path,
        image_column=args.image_column,
        label_column=args.label_column,
        valid_column=args.valid_column,
        valid_value=args.valid_value,
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    materialized = status_count(summary, MATERIALIZED_STATUSES)
    missing = summary.get("missing_source", 0)
    ignored = summary.get("ignored_unvalidated", 0)
    duplicates = summary.get("duplicate", 0)
    skipped = summary.get("skipped_existing", 0)
    print(
        f"Processed {summary['total']} rows; "
        f"{materialized} validated images selected; "
        f"{ignored} ignored; "
        f"{missing} missing; "
        f"{duplicates} duplicates; "
        f"{skipped} existing skipped."
    )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
