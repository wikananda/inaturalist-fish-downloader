import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inaturalist_downloader.dataset.final_builder import (
    build_final_dataset,
    write_records_tsv,
)


class FinalDatasetBuilderTests(unittest.TestCase):
    def test_copies_species_meeting_threshold_and_reports_excluded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "filtered_ds"
            output_dir = root / "final_ds"
            self._species(images_dir / "enough_fish", 2)
            self._species(images_dir / "low_fish", 1)

            result = build_final_dataset(images_dir, output_dir, 2)

            self.assertEqual(
                [record.species_slug for record in result.included], ["enough_fish"]
            )
            self.assertEqual(
                [record.species_slug for record in result.excluded], ["low_fish"]
            )
            self.assertEqual(result.included[0].status, "included")
            self.assertEqual(result.excluded[0].status, "excluded_below_threshold")
            self.assertEqual(result.copied_count, 1)
            self.assertTrue((output_dir / "enough_fish" / "image_0.jpg").exists())
            self.assertFalse((output_dir / "low_fish").exists())

    def test_dry_run_writes_reports_without_copying(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "filtered_ds"
            output_dir = root / "final_ds"
            included_report = root / "included.tsv"
            excluded_report = root / "excluded.tsv"
            self._species(images_dir / "enough_fish", 2)
            self._species(images_dir / "low_fish", 1)

            result = build_final_dataset(images_dir, output_dir, 2, dry_run=True)
            write_records_tsv(result.included, included_report)
            write_records_tsv(result.excluded, excluded_report)
            included_rows = self._read_tsv(included_report)
            excluded_rows = self._read_tsv(excluded_report)

            self.assertFalse(output_dir.exists())
            self.assertEqual(included_rows[0]["status"], "would_include")
            self.assertEqual(included_rows[0]["image_count"], "2")
            self.assertEqual(
                excluded_rows[0]["status"], "would_exclude_below_threshold"
            )
            self.assertEqual(excluded_rows[0]["image_count"], "1")

    def test_existing_output_is_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "filtered_ds"
            output_dir = root / "final_ds"
            self._species(images_dir / "enough_fish", 2)
            existing_dir = output_dir / "enough_fish"
            existing_dir.mkdir(parents=True)
            marker = existing_dir / "keep.txt"
            marker.write_text("existing\n", encoding="utf-8")

            result = build_final_dataset(images_dir, output_dir, 2)

            self.assertEqual(result.included[0].status, "skipped_existing")
            self.assertEqual(result.copied_count, 0)
            self.assertEqual(result.skipped_existing_count, 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing\n")

    def test_overwrite_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "filtered_ds"
            output_dir = root / "final_ds"
            self._species(images_dir / "enough_fish", 2)
            existing_dir = output_dir / "enough_fish"
            existing_dir.mkdir(parents=True)
            marker = existing_dir / "remove.txt"
            marker.write_text("old\n", encoding="utf-8")

            result = build_final_dataset(images_dir, output_dir, 2, overwrite=True)

            self.assertEqual(result.included[0].status, "included")
            self.assertEqual(result.copied_count, 1)
            self.assertFalse(marker.exists())
            self.assertTrue((output_dir / "enough_fish" / "image_1.jpg").exists())

    def _species(self, folder: Path, count: int):
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (8, 8), color="white").save(folder / f"image_{index}.jpg")

    def _read_tsv(self, path: Path):
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file, delimiter="\t"))


if __name__ == "__main__":
    unittest.main()
