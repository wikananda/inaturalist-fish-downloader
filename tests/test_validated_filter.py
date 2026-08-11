import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inaturalist_downloader.dataset.validated_filter import (
    extract_validated_images,
    resolve_source_path,
)


class ValidatedFilterTests(unittest.TestCase):
    def test_extract_validated_images_copies_only_validated_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads"
            output_dir = root / "filtered_ds"
            report_path = root / "manifests" / "validated_filter_report.jsonl"
            species_dir = images_dir / "test_fish"
            species_dir.mkdir(parents=True)
            valid_image = species_dir / "valid.jpg"
            invalid_image = species_dir / "invalid.jpg"
            self._image(valid_image, color="green")
            self._image(invalid_image, color="red")

            csv_path = root / "dataset.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "image": self._label_studio_url(valid_image),
                        "label": "test_fish",
                        "valid": "validated",
                    },
                    {
                        "image": self._label_studio_url(invalid_image),
                        "label": "test_fish",
                        "valid": "unvalidated",
                    },
                ],
            )

            summary = extract_validated_images(
                csv_path=csv_path,
                images_dir=images_dir,
                output_dir=output_dir,
                report_path=report_path,
            )
            records = self._read_jsonl(report_path)
            valid_exists = (output_dir / "test_fish" / "valid.jpg").exists()
            invalid_exists = (output_dir / "test_fish" / "invalid.jpg").exists()

        self.assertEqual(summary["copied"], 1)
        self.assertEqual(summary["ignored_unvalidated"], 1)
        self.assertTrue(valid_exists)
        self.assertFalse(invalid_exists)
        self.assertEqual([record["status"] for record in records], ["copied", "ignored_unvalidated"])

    def test_resolve_source_path_handles_label_studio_local_files_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "downloads"
            source = images_dir / "test_fish" / "fish.jpeg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"image")

            resolved = resolve_source_path(
                self._label_studio_url(source),
                label="test_fish",
                images_dir=images_dir,
            )

        self.assertEqual(resolved, source)

    def test_missing_source_and_duplicate_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads"
            output_dir = root / "filtered_ds"
            source = images_dir / "test_fish" / "fish.jpg"
            source.parent.mkdir(parents=True)
            self._image(source)
            csv_path = root / "dataset.csv"
            report_path = root / "report.jsonl"
            self._write_csv(
                csv_path,
                [
                    {
                        "image": self._label_studio_url(source),
                        "label": "test_fish",
                        "valid": "validated",
                    },
                    {
                        "image": self._label_studio_url(source),
                        "label": "test_fish",
                        "valid": "validated",
                    },
                    {
                        "image": "https://example.test/data/local-files/?d=label-studio-data/downloads/test_fish/missing.jpg",
                        "label": "test_fish",
                        "valid": "validated",
                    },
                ],
            )

            summary = extract_validated_images(
                csv_path=csv_path,
                images_dir=images_dir,
                output_dir=output_dir,
                report_path=report_path,
            )

        self.assertEqual(summary["copied"], 1)
        self.assertEqual(summary["duplicate"], 1)
        self.assertEqual(summary["missing_source"], 1)

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads"
            source = images_dir / "test_fish" / "fish.jpg"
            source.parent.mkdir(parents=True)
            self._image(source)
            csv_path = root / "dataset.csv"
            output_dir = root / "filtered_ds"
            self._write_csv(
                csv_path,
                [
                    {
                        "image": self._label_studio_url(source),
                        "label": "test_fish",
                        "valid": "validated",
                    }
                ],
            )

            summary = extract_validated_images(
                csv_path=csv_path,
                images_dir=images_dir,
                output_dir=output_dir,
                report_path=root / "report.jsonl",
                dry_run=True,
            )

        self.assertEqual(summary["would_copy"], 1)
        self.assertFalse((output_dir / "test_fish" / "fish.jpg").exists())

    def test_existing_output_is_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads"
            output_dir = root / "filtered_ds"
            source = images_dir / "test_fish" / "fish.jpg"
            existing = output_dir / "test_fish" / "fish.jpg"
            source.parent.mkdir(parents=True)
            existing.parent.mkdir(parents=True)
            self._image(source, color="green")
            existing.write_bytes(b"existing")
            csv_path = root / "dataset.csv"
            self._write_csv(
                csv_path,
                [
                    {
                        "image": self._label_studio_url(source),
                        "label": "test_fish",
                        "valid": "validated",
                    }
                ],
            )

            summary = extract_validated_images(
                csv_path=csv_path,
                images_dir=images_dir,
                output_dir=output_dir,
                report_path=root / "report.jsonl",
            )
            existing_bytes = existing.read_bytes()

        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(existing_bytes, b"existing")

    def _image(self, path: Path, color="white"):
        Image.new("RGB", (8, 8), color=color).save(path)

    def _write_csv(self, path: Path, rows: list[dict[str, str]]):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "label", "valid"])
            writer.writeheader()
            writer.writerows(rows)

    def _read_jsonl(self, path: Path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _label_studio_url(self, path: Path):
        return (
            "https://label.example.test/data/local-files/"
            f"?d=label-studio-data/downloads/{path.parent.name}/{path.name}"
        )


if __name__ == "__main__":
    unittest.main()
