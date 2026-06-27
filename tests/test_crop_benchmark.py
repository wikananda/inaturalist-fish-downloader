import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inaturalist_downloader.download.crop_benchmark import (
    box_area_ratio,
    box_iou,
    load_benchmark_samples,
    padded_crop_box,
)
from inaturalist_downloader.download.detection import select_sam3_instances


class CropBenchmarkTests(unittest.TestCase):
    def test_box_metrics(self):
        self.assertEqual(padded_crop_box((10, 10, 30, 30), width=40, height=40, padding=0.5), (0, 0, 40, 40))
        self.assertAlmostEqual(box_area_ratio((0, 0, 10, 10), 20, 20), 0.25)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertIsNone(box_iou((0, 0, 0, 0), (1, 1, 1, 1)))

    def test_load_samples_from_manifest_balances_by_species(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "candidates.jsonl"
            raw_dir = root / "raw"
            records = []
            for species in ("Alpha fish", "Beta fish"):
                folder = raw_dir / species.lower().replace(" ", "_")
                folder.mkdir(parents=True)
                for index in range(3):
                    image_path = folder / f"{index}.jpg"
                    Image.new("RGB", (16, 16), color=(index, index, index)).save(image_path)
                    records.append(
                        {
                            "canonical_name": species,
                            "raw_path": str(image_path),
                            "observation_id": index,
                            "photo_id": index,
                        }
                    )
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            samples = load_benchmark_samples(
                manifest_path=manifest,
                raw_dir=raw_dir,
                max_images=3,
                max_per_species=2,
            )

        self.assertEqual(len(samples), 3)
        self.assertEqual([sample.species for sample in samples], ["Alpha fish", "Alpha fish", "Beta fish"])

    def test_load_samples_falls_back_to_raw_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "fish_species"
            raw_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), color="blue").save(raw_dir / "one.jpg")

            samples = load_benchmark_samples(
                manifest_path=root / "missing.jsonl",
                raw_dir=root / "raw",
                max_images=5,
                max_per_species=5,
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].species, "fish_species")

    def test_select_sam3_instances_filters_and_sorts(self):
        instances = select_sam3_instances(
            masks=[
                [[1, 1], [0, 0]],
                [[1, 1], [1, 1]],
                [[0, 0], [0, 0]],
            ],
            boxes=[
                [10, 10, 20, 20],
                [30, 30, 50, 50],
                [0, 0, 5, 5],
            ],
            scores=[0.5, 0.9, 0.99],
            width=100,
            height=100,
            score_threshold=0.4,
            min_mask_area_ratio=0.1,
            crop_padding=0.1,
        )

        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]["score"], 0.9)
        self.assertEqual(instances[0]["crop_box_xyxy"], (28, 28, 52, 52))


if __name__ == "__main__":
    unittest.main()
