import argparse
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from inaturalist_downloader.download import detection as detection_module
from inaturalist_downloader.download.clip_filter import (
    _cached_snapshot_path,
    run_clip_filter_batch,
)
from inaturalist_downloader.download.crop_quality import (
    crop_geometry_metrics,
    evaluate_crop_quality,
    evaluate_redetection_quality,
)
from inaturalist_downloader.download.detection import run_fish_detection


def quality_args(**overrides):
    values = {
        "enable_crop_quality": True,
        "crop_min_short_side": 96,
        "crop_min_long_side": 160,
        "min_fish_bbox_width": 64,
        "min_fish_bbox_height": 32,
        "min_fish_crop_area_ratio": 0.20,
        "max_fish_crop_area_ratio": 0.90,
        "min_source_edge_margin_ratio": 0.003,
        "min_crop_edge_margin_ratio": 0.01,
        "min_crop_edge_variance": 8.0,
        "min_crop_entropy": 2.5,
        "sam_min_mask_crop_area_ratio": 0.12,
        "sam_max_mask_crop_area_ratio": 0.85,
        "crop_redetect": True,
        "crop_redetect_confidence": 0.35,
        "crop_redetect_require_single": True,
        "crop_redetect_min_area_ratio": 0.18,
        "crop_redetect_max_area_ratio": 0.90,
        "crop_redetect_min_iou": 0.35,
        "crop_redetect_min_edge_margin_ratio": 0.005,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def textured_image(size=(760, 420)):
    image = Image.new("RGB", size, color=(40, 90, 130))
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 16):
        draw.line((x, 0, x, size[1]), fill=(150, 180, 80), width=3)
    for y in range(0, size[1], 19):
        draw.line((0, y, size[0], y), fill=(20, 50, 100), width=2)
    return image


class CropQualityTests(unittest.TestCase):
    def test_cached_snapshot_path_resolves_huggingface_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            model_cache = cache / "models--google--siglip2-base-patch16-384"
            revision = "abc123"
            (model_cache / "refs").mkdir(parents=True)
            (model_cache / "refs" / "main").write_text(revision, encoding="utf-8")
            snapshot = model_cache / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            resolved = _cached_snapshot_path(
                "google/siglip2-base-patch16-384", str(cache)
            )

        self.assertEqual(resolved, snapshot)

    def test_good_crop_passes_and_records_geometry(self):
        args = quality_args()
        crop = textured_image()
        passed, reason, metrics = evaluate_crop_quality(
            crop=crop,
            source_size=(1000, 600),
            fish_box=(200, 150, 800, 450),
            crop_box=(120, 90, 880, 510),
            args=args,
            mask_area_ratio=0.25,
        )

        self.assertTrue(passed)
        self.assertIsNone(reason)
        self.assertGreater(metrics["geometry"]["fish_crop_area_ratio"], 0.5)
        self.assertGreater(metrics["visual"]["edge_variance"], 8.0)
        self.assertGreater(metrics["mask_crop_area_ratio"], 0.12)

    def test_source_edge_truncation_is_rejected(self):
        passed, reason, metrics = evaluate_crop_quality(
            crop=textured_image((700, 420)),
            source_size=(1000, 600),
            fish_box=(0, 150, 600, 450),
            crop_box=(0, 90, 700, 510),
            args=quality_args(),
        )

        self.assertFalse(passed)
        self.assertEqual(reason, "fish_touches_source_edge")
        self.assertEqual(metrics["geometry"]["source_min_edge_margin_ratio"], 0.0)

    def test_flat_unrecognizable_crop_is_rejected(self):
        passed, reason, _ = evaluate_crop_quality(
            crop=Image.new("RGB", (760, 420), color=(100, 100, 100)),
            source_size=(1000, 600),
            fish_box=(200, 150, 800, 450),
            crop_box=(120, 90, 880, 510),
            args=quality_args(),
        )

        self.assertFalse(passed)
        self.assertIn(reason, {"crop_too_blurry_or_flat", "crop_information_too_low"})

    def test_sam_mask_must_fill_enough_of_crop(self):
        passed, reason, metrics = evaluate_crop_quality(
            crop=textured_image(),
            source_size=(1000, 600),
            fish_box=(200, 150, 800, 450),
            crop_box=(120, 90, 880, 510),
            args=quality_args(),
            mask_area_ratio=0.01,
        )

        self.assertFalse(passed)
        self.assertEqual(reason, "sam_mask_too_small_in_crop")
        self.assertLess(metrics["mask_crop_area_ratio"], 0.12)

    def test_redetection_rejects_multiple_fish(self):
        boxes = [
            {"bbox_xyxy": [80, 40, 480, 240], "confidence": 0.9, "area_ratio": 0.51},
            {"bbox_xyxy": [10, 10, 100, 80], "confidence": 0.8, "area_ratio": 0.04},
        ]
        passed, reason, metrics = evaluate_redetection_quality(
            boxes=boxes,
            expected_box=(80, 40, 480, 240),
            crop_size=(560, 280),
            args=quality_args(),
        )

        self.assertFalse(passed)
        self.assertEqual(reason, "crop_redetect_multiple_fish")
        self.assertEqual(metrics["detection_count"], 2)

    def test_redetection_rejects_disagreement(self):
        boxes = [
            {"bbox_xyxy": [10, 10, 150, 100], "confidence": 0.9, "area_ratio": 0.08}
        ]
        passed, reason, metrics = evaluate_redetection_quality(
            boxes=boxes,
            expected_box=(80, 40, 480, 240),
            crop_size=(560, 280),
            args=quality_args(crop_redetect_min_area_ratio=0.0),
        )

        self.assertFalse(passed)
        self.assertEqual(reason, "crop_redetect_box_disagreement")
        self.assertLess(metrics["agreement_iou"], 0.35)

    def test_yolo_path_applies_quality_and_redetection_before_saving(self):
        source_box = {
            "bbox_xyxy": [100, 100, 500, 300],
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "fish",
            "area_ratio": 0.5,
            "selection_score": 0.45,
        }
        redetected_box = {
            "bbox_xyxy": [80, 40, 480, 240],
            "confidence": 0.91,
            "class_id": 0,
            "class_name": "fish",
            "area_ratio": 0.51,
            "selection_score": 0.46,
        }
        args = quality_args(
            overwrite=True,
            allow_multiple_fish=False,
            min_fish_area_ratio=0.02,
            crop_padding=0.20,
            detector_weights="models/fish-yolo.pt",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.jpg"
            accepted_path = root / "accepted.jpg"
            textured_image((600, 400)).save(raw_path)
            with patch.object(
                detection_module,
                "_yolo_detect_boxes",
                side_effect=[([source_box], {"stage": "source"}), ([redetected_box], {"stage": "crop"})],
            ):
                passed, reason, metrics = run_fish_detection(raw_path, accepted_path, args)

            self.assertTrue(passed)
            self.assertIsNone(reason)
            self.assertTrue(accepted_path.exists())
            self.assertTrue(metrics["crop_quality"]["passed"])
            self.assertTrue(metrics["crop_redetection"]["passed"])

    def test_siglip_probability_margin_rejects_bad_crop(self):
        class FakeInput:
            def to(self, device):
                return self

        class FakeProcessor:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"pixel_values": FakeInput(), "input_ids": FakeInput()}

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class FakeModel:
            def to(self, device):
                return self

            def eval(self):
                return None

            def __call__(self, **inputs):
                # First image: positive wins. Second image: negative wins.
                return types.SimpleNamespace(
                    logits_per_image=FakeTensor([[2.0, -2.0], [-2.0, 2.0]])
                )

        class NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        def sigmoid(tensor):
            return FakeTensor(
                [
                    [1.0 / (1.0 + math.exp(-value)) for value in row]
                    for row in tensor.values
                ]
            )

        fake_torch = types.SimpleNamespace(
            no_grad=Mock(return_value=NoGrad()), sigmoid=sigmoid
        )
        args = argparse.Namespace(
            clip_backend="siglip2",
            clip_model="google/siglip2-base-patch16-384",
            clip_cache_dir="models",
            clip_device="cpu",
            clip_positive_prompts=["a complete fish"],
            clip_negative_prompts=["a partial fish"],
            clip_threshold=0.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for index in range(2):
                path = Path(temp_dir) / f"{index}.jpg"
                textured_image((64, 64)).save(path)
                paths.append(path)
            with patch.dict(sys.modules, {"torch": fake_torch}), patch(
                "inaturalist_downloader.download.clip_filter.get_clip_components",
                return_value=(FakeModel(), FakeProcessor()),
            ), patch(
                "inaturalist_downloader.download.clip_filter.resolve_clip_device",
                return_value="cpu",
            ):
                results = run_clip_filter_batch(paths, args)

        self.assertTrue(results[0][0])
        self.assertFalse(results[1][0])
        self.assertEqual(results[0][2]["backend"], "siglip2")
        self.assertEqual(
            results[0][2]["score_kind"], "sigmoid_probability_margin"
        )


if __name__ == "__main__":
    unittest.main()
