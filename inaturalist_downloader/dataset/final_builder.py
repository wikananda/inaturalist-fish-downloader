"""Build a final dataset from species folders that meet a minimum image count."""

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from .checks import count_images, species_name_from_folder_slug


@dataclass(frozen=True)
class FinalDatasetRecord:
    """One species decision for the final dataset build."""

    species_slug: str
    species_name: str
    image_count: int
    threshold: int
    status: str

    def as_row(self) -> dict[str, str]:
        """Return this record as a TSV row."""
        return {
            "species_slug": self.species_slug,
            "species_name": self.species_name,
            "image_count": str(self.image_count),
            "threshold": str(self.threshold),
            "status": self.status,
        }


@dataclass(frozen=True)
class FinalDatasetResult:
    """Summary of a final dataset build."""

    included: list[FinalDatasetRecord]
    excluded: list[FinalDatasetRecord]
    copied_count: int
    skipped_existing_count: int


def build_final_dataset(
    images_dir: Path,
    output_dir: Path,
    threshold: int,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> FinalDatasetResult:
    """Copy species folders meeting threshold from images_dir into output_dir."""
    if threshold < 1:
        raise ValueError("threshold must be 1 or greater")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    included: list[FinalDatasetRecord] = []
    excluded: list[FinalDatasetRecord] = []
    copied_count = 0
    skipped_existing_count = 0

    species_folders = sorted(path for path in images_dir.iterdir() if path.is_dir())
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for source_dir in species_folders:
        image_count = count_images(source_dir)
        species_slug = source_dir.name
        species_name = species_name_from_folder_slug(species_slug)

        if image_count < threshold:
            status = (
                "would_exclude_below_threshold"
                if dry_run
                else "excluded_below_threshold"
            )
            excluded.append(
                FinalDatasetRecord(
                    species_slug=species_slug,
                    species_name=species_name,
                    image_count=image_count,
                    threshold=threshold,
                    status=status,
                )
            )
            continue

        destination_dir = output_dir / species_slug
        status = "would_include" if dry_run else "included"
        if destination_dir.exists() and not overwrite:
            status = "skipped_existing"
            skipped_existing_count += 1
        elif not dry_run:
            if destination_dir.exists():
                shutil.rmtree(destination_dir)
            shutil.copytree(source_dir, destination_dir)
            copied_count += 1

        included.append(
            FinalDatasetRecord(
                species_slug=species_slug,
                species_name=species_name,
                image_count=image_count,
                threshold=threshold,
                status=status,
            )
        )

    return FinalDatasetResult(
        included=included,
        excluded=excluded,
        copied_count=copied_count,
        skipped_existing_count=skipped_existing_count,
    )


def write_records_tsv(records: list[FinalDatasetRecord], path: Path) -> None:
    """Write final dataset records to a TSV report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["species_slug", "species_name", "image_count", "threshold", "status"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())
