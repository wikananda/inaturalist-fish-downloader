import argparse
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from inaturalist_downloader.common.inat import iter_observation_photos
from inaturalist_downloader.commands.download import download_species_images
from inaturalist_downloader.download.cli import merge_filter_configs, parse_args, validate_args
from inaturalist_downloader.download import clip_filter as clip_module
from inaturalist_downloader.download import detection as detection_module
from inaturalist_downloader.download.clip_filter import preload_clip_model
from inaturalist_downloader.download.detection import (
    DetectionOutput,
    _box_iou,
    _disable_sam_internal_bf16_contexts,
    _resolve_sam_precision,
    _yolo_box_to_sam_prompt,
    ensure_sam3_model_files,
    get_sam3_model,
    run_cascade_detection_outputs,
    run_fish_detection_outputs,
    run_sam3_detection_outputs,
)


class DownloadFilterConfigTests(unittest.TestCase):
    def test_profile_filter_file_merges_before_profile_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filter_path = temp_path / "alive.yaml"
            profile_path = temp_path / "profile.yaml"
            filter_path.write_text(
                "inat:\n"
                "  alive_only: true\n"
                "  order_by: created_at\n"
                "  query_params:\n"
                "    licensed: true\n",
                encoding="utf-8",
            )
            profile_path.write_text(
                "inat:\n"
                "  filter_files:\n"
                "    - alive.yaml\n"
                "  order_by: votes\n",
                encoding="utf-8",
            )

            argv = ["inat-download", "--config", str(profile_path)]
            with patch.object(sys, "argv", argv):
                args = parse_args()

        self.assertTrue(args.alive_only)
        self.assertEqual(args.order_by, "votes")
        self.assertEqual(args.query_params, {"licensed": True})

    def test_filter_query_param_exclusions_are_concatenated(self):
        juvenile_filter = OmegaConf.create(
            {
                "inat": {
                    "query_params": {
                        "without_term_id": 1,
                        "without_term_value_id": 8,
                    }
                }
            }
        )
        female_filter = OmegaConf.create(
            {
                "inat": {
                    "query_params": {
                        "without_term_id": 9,
                        "without_term_value_id": 10,
                    }
                }
            }
        )

        merged = merge_filter_configs([juvenile_filter, female_filter])
        query_params = OmegaConf.to_container(merged, resolve=True)["inat"]["query_params"]

        self.assertEqual(query_params["without_term_id"], [1, 9])
        self.assertEqual(query_params["without_term_value_id"], [8, 10])

    def test_iter_observation_photos_includes_normalized_raw_query_params(self):
        payload = {
            "results": [
                {
                    "id": 123,
                    "quality_grade": "research",
                    "photos": [{"id": 456, "url": "https://example.test/square.jpg"}],
                    "user": {"id": 789, "login": "observer"},
                }
            ]
        }

        with patch("inaturalist_downloader.common.inat.api_get", return_value=payload) as api_get:
            photos = list(
                iter_observation_photos(
                    taxon_id=1,
                    quality_grade="any",
                    per_page=10,
                    max_pages=1,
                    license_code=None,
                    place_id=None,
                    exclude_captive=False,
                    term_id=None,
                    term_value_id=None,
                    order_by="created_at",
                    order="asc",
                    query_params={
                        "licensed": True,
                        "photo_license": ["cc-by", "cc-by-nc"],
                        "ignored": None,
                    },
                )
            )

        self.assertEqual(len(photos), 1)
        api_get.assert_called_once()
        params = api_get.call_args.kwargs
        self.assertEqual(params["order_by"], "created_at")
        self.assertEqual(params["order"], "asc")
        self.assertEqual(params["licensed"], "true")
        self.assertEqual(params["photo_license"], "cc-by,cc-by-nc")
        self.assertNotIn("ignored", params)

    def test_ensure_sam3_model_files_uses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint = temp_path / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            args = argparse.Namespace(
                sam_checkpoint_path=str(checkpoint),
                sam_model_dir=str(temp_path / "models"),
                sam_checkpoint_filename="sam3.1_multiplex.pt",
            )
            fake_hub = types.SimpleNamespace(hf_hub_download=Mock())

            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                resolved = ensure_sam3_model_files(args)

        self.assertEqual(resolved, checkpoint)
        self.assertEqual(args.sam_checkpoint_path, str(checkpoint))
        fake_hub.hf_hub_download.assert_not_called()

    def test_ensure_sam3_model_files_downloads_config_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models" / "sam3.1"
            checkpoint = model_dir / "sam3.1_multiplex.pt"
            args = argparse.Namespace(
                sam_checkpoint_path=None,
                sam_model_dir=str(model_dir),
                sam_repo_id="facebook/sam3.1",
                sam_config_filename="config.json",
                sam_checkpoint_filename="sam3.1_multiplex.pt",
            )

            def hf_hub_download(repo_id, filename, local_dir):
                path = Path(local_dir) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"model")
                return str(path)

            fake_hub = types.SimpleNamespace(
                hf_hub_download=Mock(side_effect=hf_hub_download)
            )

            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                resolved = ensure_sam3_model_files(args)

        self.assertEqual(resolved, checkpoint)
        self.assertEqual(args.sam_checkpoint_path, str(checkpoint))
        self.assertEqual(
            [call.kwargs["filename"] for call in fake_hub.hf_hub_download.call_args_list],
            ["config.json", "sam3.1_multiplex.pt"],
        )

    def test_validate_args_rejects_invalid_sam_dtype(self):
        args = self._validation_args(enable_detection=True)
        args.detection_backend = "sam3"
        args.sam_dtype = "bf16"

        with self.assertRaisesRegex(SystemExit, "--sam-dtype must be one of"):
            validate_args(args)

    def test_get_sam3_model_cache_includes_dtype_and_autocast(self):
        class FakeModel:
            def __init__(self, context=None):
                self.to_calls = []
                self.eval_called = False
                self.bf16_context = context

            def to(self, **kwargs):
                self.to_calls.append(kwargs)
                return self

            def eval(self):
                self.eval_called = True

            def modules(self):
                return [self]

        class FakeContext:
            def __init__(self):
                self.exit_called = False

            def __exit__(self, exc_type, exc, traceback):
                self.exit_called = True
                return False

        fake_torch = types.SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        )
        built = []
        contexts = []

        def builder(checkpoint_path=None, device=None):
            context = FakeContext()
            contexts.append(context)
            model = FakeModel(context)
            built.append((checkpoint_path, device, model))
            return model

        self._reset_sam3_model_cache()
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}):
                first = get_sam3_model(
                    builder,
                    "cuda",
                    "models/sam3.1/sam3.1_multiplex.pt",
                    "float32",
                    False,
                )
                second = get_sam3_model(
                    builder,
                    "cuda",
                    "models/sam3.1/sam3.1_multiplex.pt",
                    "float32",
                    False,
                )
                third = get_sam3_model(
                    builder,
                    "cuda",
                    "models/sam3.1/sam3.1_multiplex.pt",
                    "float32",
                    True,
                )
        finally:
            self._reset_sam3_model_cache()

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(built), 2)
        self.assertEqual(built[0][0], "models/sam3.1/sam3.1_multiplex.pt")
        self.assertEqual(built[0][1], "cuda")
        self.assertEqual(first.to_calls[0], {"device": "cuda", "dtype": "float32"})
        self.assertTrue(first.eval_called)
        self.assertTrue(contexts[0].exit_called)
        self.assertFalse(contexts[1].exit_called)

    def test_disable_sam_internal_bf16_contexts_exits_module_contexts(self):
        class FakeContext:
            def __init__(self):
                self.exit_called = False

            def __exit__(self, exc_type, exc, traceback):
                self.exit_called = True
                return False

        class FakeModule:
            def __init__(self, context=None):
                self.bf16_context = context

        class FakeModel:
            def __init__(self):
                self.first_context = FakeContext()
                self.second_context = FakeContext()
                self.first = FakeModule(self.first_context)
                self.second = FakeModule(self.second_context)

            def modules(self):
                return [self.first, self.second]

        model = FakeModel()
        disabled = _disable_sam_internal_bf16_contexts(model)

        self.assertEqual(disabled, 2)
        self.assertTrue(model.first_context.exit_called)
        self.assertTrue(model.second_context.exit_called)

    def test_resolve_sam_precision_keeps_bf16_autocast_on_cuda(self):
        self.assertEqual(
            _resolve_sam_precision("cuda", "bfloat16", True), ("bfloat16", True)
        )

    def test_resolve_sam_precision_falls_back_off_cuda(self):
        # bf16 autocast is CUDA-only; MPS/CPU fall back to plain float32.
        self.assertEqual(
            _resolve_sam_precision("mps", "bfloat16", True), ("float32", False)
        )
        self.assertEqual(
            _resolve_sam_precision("cpu", "bfloat16", True), ("float32", False)
        )

    def test_resolve_sam_precision_leaves_explicit_float32_untouched(self):
        self.assertEqual(
            _resolve_sam_precision("cuda", "float32", False), ("float32", False)
        )

    def test_sam3_detection_metrics_include_dtype_autocast_and_device(self):
        class FakeAutocast:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeModel:
            def to(self, **kwargs):
                return self

            def eval(self):
                return None

        captured_thresholds = []

        class FakeProcessor:
            def __init__(self, model, device="cuda", confidence_threshold=0.5):
                self.device = device
                self.confidence_threshold = confidence_threshold
                captured_thresholds.append(confidence_threshold)

            def set_image(self, image):
                return {}

            def set_text_prompt(self, state, prompt):
                return {
                    "boxes": [[1.0, 1.0, 20.0, 20.0]],
                    "scores": [0.9],
                    "masks": None,
                }

        fake_torch = types.SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            autocast=Mock(return_value=FakeAutocast()),
        )
        processor_module = types.ModuleType("sam3.model.sam3_image_processor")
        processor_module.Sam3Processor = FakeProcessor
        builder_module = types.ModuleType("sam3.model_builder")
        builder_module.build_sam3_image_model = Mock(return_value=FakeModel())

        with tempfile.TemporaryDirectory() as temp_dir:
            from PIL import Image

            temp_path = Path(temp_dir)
            raw_path = temp_path / "raw.jpg"
            accepted_path = temp_path / "accepted.jpg"
            Image.new("RGB", (32, 32), color="blue").save(raw_path)
            args = argparse.Namespace(
                detector_device="cpu",
                sam_repo_id="facebook/sam3.1",
                sam_checkpoint_path="models/sam3.1/sam3.1_multiplex.pt",
                sam_dtype="float32",
                sam_autocast=False,
                sam_prompt="fish",
                sam_crop_padding=0.0,
                sam_score_threshold=0.3,
                sam_min_mask_area_ratio=0.0,
                sam_max_instances_per_image=None,
                sam_save_all_instances=True,
                overwrite=True,
            )

            self._reset_sam3_model_cache()
            try:
                with patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "sam3": types.ModuleType("sam3"),
                        "sam3.model": types.ModuleType("sam3.model"),
                        "sam3.model.sam3_image_processor": processor_module,
                        "sam3.model_builder": builder_module,
                    },
                ):
                    outputs, reject_reason, metrics = run_sam3_detection_outputs(
                        raw_path,
                        accepted_path,
                        args,
                    )
            finally:
                self._reset_sam3_model_cache()

        self.assertIsNone(reject_reason)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(metrics["dtype"], "float32")
        self.assertFalse(metrics["autocast"])
        self.assertEqual(metrics["device"], "cpu")
        builder_module.build_sam3_image_model.assert_called_once_with(
            checkpoint_path="models/sam3.1/sam3.1_multiplex.pt",
            device="cpu",
        )
        fake_torch.autocast.assert_called_once_with(
            device_type="cpu",
            enabled=False,
        )
        # The pipeline threshold must reach the processor's internal confidence gate,
        # otherwise SAM 3 silently filters everything at its 0.5 default.
        self.assertEqual(captured_thresholds, [0.3])
        self.assertEqual(metrics["confidence_threshold"], 0.3)

    def test_cascade_helpers_box_conversion_and_iou(self):
        # pixel [x1,y1,x2,y2] -> normalized [cx,cy,w,h]
        self.assertEqual(
            _yolo_box_to_sam_prompt([0.0, 0.0, 50.0, 100.0], 100, 100),
            [0.25, 0.5, 0.5, 1.0],
        )
        # out-of-range values are clamped to [0, 1]
        self.assertEqual(
            _yolo_box_to_sam_prompt([-10.0, -10.0, 200.0, 200.0], 100, 100),
            [0.95, 0.95, 1.0, 1.0],
        )
        self.assertEqual(_box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(_box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def _run_cascade(self, yolo_boxes, sam_output):
        """Helper: run run_cascade_detection_outputs with YOLO + SAM faked out."""

        class FakeModel:
            def to(self, **kwargs):
                return self

            def eval(self):
                return None

            def modules(self):
                return [self]

        class FakeProcessor:
            def __init__(self, model, device="cuda", confidence_threshold=0.5):
                self.device = device
                self.reset_calls = 0

            def set_image(self, image):
                return {"image_set": True}

            def reset_all_prompts(self, state):
                self.reset_calls += 1
                return state

            def add_geometric_prompt(self, box, label, state):
                return sam_output

        class FakeAutocast:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_torch = types.SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            autocast=Mock(return_value=FakeAutocast()),
        )
        processor_module = types.ModuleType("sam3.model.sam3_image_processor")
        processor_module.Sam3Processor = FakeProcessor
        builder_module = types.ModuleType("sam3.model_builder")
        builder_module.build_sam3_image_model = Mock(return_value=FakeModel())

        def fake_yolo(image, args):
            return list(yolo_boxes), {"enabled": True, "fish_detection_count": len(yolo_boxes)}

        from PIL import Image

        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        temp_path = Path(temp_dir)
        raw_path = temp_path / "raw.jpg"
        accepted_path = temp_path / "accepted.jpg"
        Image.new("RGB", (64, 64), color="blue").save(raw_path)
        args = argparse.Namespace(
            detector_weights="models/fish-yolo.pt",
            detector_device="cpu",
            sam_repo_id="facebook/sam3.1",
            sam_checkpoint_path="models/sam3.1/sam3.1_multiplex.pt",
            sam_dtype="float32",
            sam_autocast=False,
            sam_score_threshold=0.0,
            sam_min_mask_area_ratio=0.0,
            sam_crop_padding=0.0,
            sam_save_all_instances=True,
            sam_max_instances_per_image=None,
            min_fish_area_ratio=0.0,
            allow_multiple_fish=True,
            crop_padding=0.0,
            overwrite=True,
        )

        self._reset_sam3_model_cache()
        try:
            with patch.object(detection_module, "_yolo_detect_boxes", fake_yolo):
                with patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "sam3": types.ModuleType("sam3"),
                        "sam3.model": types.ModuleType("sam3.model"),
                        "sam3.model.sam3_image_processor": processor_module,
                        "sam3.model_builder": builder_module,
                    },
                ):
                    outputs, reject_reason, metrics = run_cascade_detection_outputs(
                        raw_path, accepted_path, args
                    )
        finally:
            self._reset_sam3_model_cache()
        return outputs, reject_reason, metrics, accepted_path

    def test_cascade_refines_each_yolo_box(self):
        yolo_boxes = [
            {
                "bbox_xyxy": [5.0, 5.0, 25.0, 25.0],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "fish",
                "area_ratio": 0.3,
                "selection_score": 0.27,
            },
            {
                "bbox_xyxy": [38.0, 38.0, 58.0, 58.0],
                "confidence": 0.7,
                "class_id": 0,
                "class_name": "fish",
                "area_ratio": 0.2,
                "selection_score": 0.14,
            },
        ]
        sam_output = {"boxes": [[10.0, 10.0, 22.0, 22.0]], "scores": [0.8], "masks": None}
        outputs, reject_reason, metrics, accepted_path = self._run_cascade(
            yolo_boxes, sam_output
        )

        self.assertIsNone(reject_reason)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(metrics["backend"], "cascade")
        self.assertEqual(metrics["yolo_fish_count"], 2)
        for index, output in enumerate(outputs, start=1):
            self.assertEqual(output.accepted_path.name, f"accepted__inst_{index}.jpg")
            self.assertEqual(output.clip_source_path, output.accepted_path)
            self.assertTrue(output.accepted_path.exists())
            self.assertEqual(
                output.metrics["selected_detection"]["refine"], "sam_refined"
            )

    def test_cascade_falls_back_to_yolo_box_when_sam_empty(self):
        yolo_boxes = [
            {
                "bbox_xyxy": [5.0, 5.0, 25.0, 25.0],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "fish",
                "area_ratio": 0.3,
                "selection_score": 0.27,
            }
        ]
        sam_output = {"boxes": [], "scores": [], "masks": None}
        outputs, reject_reason, metrics, accepted_path = self._run_cascade(
            yolo_boxes, sam_output
        )

        self.assertIsNone(reject_reason)
        self.assertEqual(len(outputs), 1)
        detection = outputs[0].metrics["selected_detection"]
        self.assertEqual(detection["refine"], "fallback_yolo")
        self.assertEqual(detection["crop_box_xyxy"], [5, 5, 25, 25])
        self.assertTrue(outputs[0].accepted_path.exists())

    def test_preload_clip_model_loads_and_warms(self):
        class FakeTensor:
            def to(self, device):
                return self

        class FakeProcessor:
            def __call__(self, text, images, return_tensors, padding):
                return {"input_ids": FakeTensor(), "pixel_values": FakeTensor()}

        class FakeModel:
            def __init__(self):
                self.eval_called = False
                self.forward_calls = 0

            def to(self, device):
                return self

            def eval(self):
                self.eval_called = True

            def __call__(self, **inputs):
                self.forward_calls += 1
                return {}

        class FakeNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_torch = types.SimpleNamespace(no_grad=Mock(return_value=FakeNoGrad()))
        fake_model = FakeModel()
        args = argparse.Namespace(
            clip_model="openai/clip-vit-base-patch32",
            clip_cache_dir="models",
            clip_device="cpu",
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch.object(clip_module, "validate_clip_import", lambda: None):
                with patch.object(
                    clip_module,
                    "get_clip_components",
                    return_value=(fake_model, FakeProcessor()),
                ):
                    with patch.object(
                        clip_module, "resolve_clip_device", return_value="cpu"
                    ):
                        result = preload_clip_model(args)

        self.assertEqual(result, "openai/clip-vit-base-patch32")
        self.assertTrue(fake_model.eval_called)
        # The warmup runs exactly one forward pass so weights are fully initialized.
        self.assertEqual(fake_model.forward_calls, 1)

    def test_cascade_dispatch_routes_to_cascade(self):
        args = argparse.Namespace(detection_backend="cascade")
        sentinel = ([], "stub", {"backend": "cascade"})
        with patch.object(
            detection_module, "run_cascade_detection_outputs", return_value=sentinel
        ) as mocked:
            result = run_fish_detection_outputs(
                Path("raw.jpg"), Path("accepted.jpg"), args, max_outputs=3
            )
        self.assertEqual(result, sentinel)
        mocked.assert_called_once()

    def test_validate_args_rejects_protected_raw_query_params(self):
        args = argparse.Namespace(
            images_per_species=1,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            species_workers=1,
            download_workers=1,
            term_value_id=None,
            term_id=None,
            alive_only=False,
            query_params={"page": 2},
            blocked_license_codes=[],
            license_preference=[],
            license_code=None,
            skip_image_validation=True,
            min_width=0,
            min_height=0,
            min_file_size_kb=0,
            max_aspect_ratio=0,
            min_intensity_range=0,
            enable_detection=False,
            enable_clip_filter=False,
        )

        with self.assertRaisesRegex(SystemExit, "protected parameters: page"):
            validate_args(args)

    def test_validate_args_rejects_blocked_license_preference(self):
        args = argparse.Namespace(
            images_per_species=1,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            species_workers=1,
            download_workers=1,
            term_value_id=None,
            term_id=None,
            alive_only=False,
            query_params={},
            blocked_license_codes=["cc-by-nc"],
            license_preference=["cc0", "cc-by-nc"],
            license_code=None,
            skip_image_validation=True,
            min_width=0,
            min_height=0,
            min_file_size_kb=0,
            max_aspect_ratio=0,
            min_intensity_range=0,
            enable_detection=False,
            enable_clip_filter=False,
        )

        with self.assertRaisesRegex(SystemExit, "blocked licenses: cc-by-nc"):
            validate_args(args)

    def test_download_species_images_falls_back_through_license_preference(self):
        args = self._download_args(images_per_species=2)
        calls = []

        def collect_photo_jobs(**kwargs):
            calls.append(kwargs["license_code"])
            if kwargs["license_code"] == "cc0":
                return [], 2, True
            return [
                self._candidate(1, "cc-by"),
                self._candidate(2, "cc-by"),
            ], 2, True

        def download_photo_job(candidate, destination, overwrite, sleep_seconds, retries):
            return {**candidate, "raw_path": str(destination), "download_status": "downloaded"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                side_effect=collect_photo_jobs,
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
                side_effect=download_photo_job,
            ), patch(
                "inaturalist_downloader.commands.download.save_accepted_image",
                return_value="accepted",
            ):
                download_species_images(
                    "Test fish",
                    args,
                    temp_path / "downloads",
                    temp_path / "raw",
                    manifest_dir,
                )

        self.assertEqual(calls, ["cc0", "cc-by"])

    def test_download_species_images_skips_blocked_license_before_download(self):
        args = self._download_args(images_per_species=1)
        args.license_preference = ["cc0"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                return_value=([self._candidate(1, "cc-by-nc")], 2, True),
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
            ) as download_photo_job:
                download_species_images(
                    "Test fish",
                    args,
                    temp_path / "downloads",
                    temp_path / "raw",
                    manifest_dir,
                )

        download_photo_job.assert_not_called()

    def test_download_species_images_accepts_multiple_sam_outputs(self):
        args = self._download_args(images_per_species=2)
        args.enable_detection = True
        args.detection_backend = "sam3"

        def download_photo_job(candidate, destination, overwrite, sleep_seconds, retries):
            return {**candidate, "raw_path": str(destination), "download_status": "downloaded"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "downloads"
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            first_output = output_dir / "test_fish" / "fish__inst_1.jpg"
            second_output = output_dir / "test_fish" / "fish__inst_2.jpg"
            detection_outputs = [
                DetectionOutput(
                    accepted_path=first_output,
                    status="accepted_crop",
                    metrics={"backend": "sam3", "instance_index": 1},
                    clip_source_path=first_output,
                    instance_index=1,
                    instance_count=2,
                    species_verification="unverified",
                    created_output=True,
                ),
                DetectionOutput(
                    accepted_path=second_output,
                    status="accepted_crop",
                    metrics={"backend": "sam3", "instance_index": 2},
                    clip_source_path=second_output,
                    instance_index=2,
                    instance_count=2,
                    species_verification="unverified",
                    created_output=True,
                ),
            ]

            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                return_value=([self._candidate(1, "cc0")], 2, True),
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
                side_effect=download_photo_job,
            ), patch(
                "inaturalist_downloader.commands.download.run_fish_detection_outputs",
                return_value=(detection_outputs, None, {"backend": "sam3"}),
            ):
                download_species_images(
                    "Test fish",
                    args,
                    output_dir,
                    temp_path / "raw",
                    manifest_dir,
                )

            accepted_lines = [
                line for line in (manifest_dir / "accepted.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(len(accepted_lines), 2)
        self.assertIn("fish__inst_1.jpg", accepted_lines[0])
        self.assertIn('"species_verification": "unverified"', accepted_lines[0])

    def _download_args(self, images_per_species=1):
        return argparse.Namespace(
            images_per_species=images_per_species,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            retries=1,
            run_id="test-run",
            include_subspecies=False,
            quality_grade="research",
            photo_size="small",
            place_id=None,
            exclude_captive=True,
            alive_only=False,
            term_id=None,
            term_value_id=None,
            order_by="votes",
            order="desc",
            query_params={},
            license_code=None,
            license_preference=["cc0", "cc-by", "cc-by-sa"],
            blocked_license_code_set={"cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd"},
            overwrite=False,
            sleep_seconds=0,
            download_workers=1,
            skip_image_validation=True,
            enable_detection=False,
            detection_backend="yolo",
            enable_clip_filter=False,
        )

    def _validation_args(self, enable_detection=False):
        return argparse.Namespace(
            images_per_species=1,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            species_workers=1,
            download_workers=1,
            term_value_id=None,
            term_id=None,
            alive_only=False,
            query_params={},
            blocked_license_codes=[],
            license_preference=[],
            license_code=None,
            skip_image_validation=True,
            min_width=0,
            min_height=0,
            min_file_size_kb=0,
            max_aspect_ratio=0,
            min_intensity_range=0,
            enable_detection=enable_detection,
            detection_backend="yolo",
            detector_weights="models/fish-yolo.pt",
            detector_confidence=0.5,
            detector_imgsz=640,
            min_fish_area_ratio=0.02,
            crop_padding=0.15,
            sam_score_threshold=0.0,
            sam_min_mask_area_ratio=0.02,
            sam_crop_padding=0.15,
            sam_dtype="float32",
            sam_autocast=False,
            sam_max_instances_per_image=None,
            detector_class_ids=None,
            detector_class_names=None,
            enable_clip_filter=False,
        )

    def _reset_sam3_model_cache(self):
        detection_module.SAM3_MODEL = None
        detection_module.SAM3_MODEL_DEVICE = None
        detection_module.SAM3_MODEL_CHECKPOINT_PATH = None
        detection_module.SAM3_MODEL_DTYPE = None
        detection_module.SAM3_MODEL_AUTOCAST = None

    def _candidate(self, photo_id, license_code):
        return {
            "run_id": "test-run",
            "species_name": "Test fish",
            "canonical_name": "Test fish",
            "taxon_id": 1,
            "observation_id": photo_id,
            "photo_id": photo_id,
            "photo_url": f"https://example.test/{photo_id}.jpg",
            "source_photo_url": f"https://example.test/{photo_id}.jpg",
            "filename": f"test_fish__obs_{photo_id}__photo_{photo_id}.jpg",
            "requested_license_code": license_code,
            "license_priority": 1,
            "license_code": license_code,
            "quality_grade": "research",
            "place_id": None,
            "observed_on": None,
            "time_observed_at": None,
            "captive": False,
            "place_guess": None,
            "user_id": None,
            "user_login": None,
            "status": "candidate",
            "reject_reason": None,
            "scores": {},
        }


if __name__ == "__main__":
    unittest.main()
