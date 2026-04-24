"""Optional YOLO fish detection and crop generation."""

import argparse
import threading
from pathlib import Path
from typing import Optional

from .image_quality import Image, ImageOps, pillow_available, save_pil_image

DETECTOR_LOCK = threading.Lock()
DETECTOR_MODEL = None
DETECTOR_MODEL_PATH = None


def get_detector_model(weights_path: str):
    """Load and cache the YOLO detector model."""
    global DETECTOR_MODEL, DETECTOR_MODEL_PATH

    with DETECTOR_LOCK:
        if DETECTOR_MODEL is not None and DETECTOR_MODEL_PATH == weights_path:
            return DETECTOR_MODEL

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(_ultralytics_error_message(exc)) from exc

        DETECTOR_MODEL = YOLO(weights_path)
        DETECTOR_MODEL_PATH = weights_path
        return DETECTOR_MODEL


def _ultralytics_error_message(exc: Exception) -> str:
    """Build an actionable Ultralytics import error message."""
    import sys

    return (
        "YOLO detection requires a working Ultralytics install in the "
        f"current Python interpreter ({sys.executable}). Original import "
        f"error: {type(exc).__name__}: {exc}"
    )


def detection_class_allowed(
    class_id: int,
    class_name: str,
    allowed_class_ids: set[int],
    allowed_class_names: set[str],
) -> bool:
    """Return whether a detector class should be treated as fish."""
    if allowed_class_ids and class_id not in allowed_class_ids:
        return False
    if allowed_class_names and class_name.casefold() not in allowed_class_names:
        return False
    return True


def validate_detector_import() -> None:
    """Fail early if Ultralytics cannot import in the active interpreter."""
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_ultralytics_error_message(exc)) from exc


def run_fish_detection(
    raw_path: Path,
    accepted_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, Optional[str], dict]:
    """Run YOLO detection, reject bad detections, and save a padded fish crop."""
    if not pillow_available():
        return False, "pillow_not_installed", {"enabled": True}

    if accepted_path.exists() and not args.overwrite:
        return True, None, {
            "enabled": True,
            "saved": "existing",
            "created_output": False,
            "model": args.detector_weights,
        }

    model = get_detector_model(args.detector_weights)
    allowed_class_ids = args.detector_class_id_set
    allowed_class_names = args.detector_class_name_set

    with Image.open(raw_path) as source_image:
        image_format = source_image.format
        image = ImageOps.exif_transpose(source_image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        width, height = image.size
        image_area = max(width * height, 1)

        predict_kwargs = {
            "source": image,
            "conf": args.detector_confidence,
            "imgsz": args.detector_imgsz,
            "verbose": False,
        }
        if args.detector_device:
            predict_kwargs["device"] = args.detector_device

        with DETECTOR_LOCK:
            results = model.predict(**predict_kwargs)

        result = results[0]
        raw_boxes = []
        if result.boxes is not None:
            xyxy_values = result.boxes.xyxy.cpu().tolist()
            conf_values = result.boxes.conf.cpu().tolist()
            cls_values = result.boxes.cls.cpu().tolist()
            names = result.names or {}

            for xyxy, confidence, class_value in zip(
                xyxy_values, conf_values, cls_values
            ):
                class_id = int(class_value)
                class_name = str(names.get(class_id, class_id))
                if not detection_class_allowed(
                    class_id,
                    class_name,
                    allowed_class_ids,
                    allowed_class_names,
                ):
                    continue

                x1, y1, x2, y2 = [float(value) for value in xyxy]
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                area_ratio = (box_width * box_height) / image_area
                raw_boxes.append(
                    {
                        "bbox_xyxy": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ],
                        "confidence": round(float(confidence), 6),
                        "class_id": class_id,
                        "class_name": class_name,
                        "area_ratio": round(area_ratio, 6),
                        "selection_score": float(confidence) * area_ratio,
                    }
                )

        metrics = {
            "enabled": True,
            "model": args.detector_weights,
            "created_output": False,
            "confidence_threshold": args.detector_confidence,
            "raw_detection_count": int(0 if result.boxes is None else len(result.boxes)),
            "fish_detection_count": len(raw_boxes),
            "min_fish_area_ratio": args.min_fish_area_ratio,
            "crop_padding": args.crop_padding,
            "allowed_class_ids": sorted(allowed_class_ids),
            "allowed_class_names": sorted(allowed_class_names),
        }

        if not raw_boxes:
            return False, "no_fish_detected", metrics

        if len(raw_boxes) > 1 and not args.allow_multiple_fish:
            metrics["detections"] = raw_boxes
            return False, "multiple_fish_detected", metrics

        selected = max(raw_boxes, key=lambda item: item["selection_score"])
        metrics["selected_detection"] = {
            key: value for key, value in selected.items() if key != "selection_score"
        }

        if selected["area_ratio"] < args.min_fish_area_ratio:
            return False, "fish_too_small", metrics

        x1, y1, x2, y2 = selected["bbox_xyxy"]
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = box_width * args.crop_padding
        pad_y = box_height * args.crop_padding
        crop_box = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
        metrics["crop_box_xyxy"] = list(crop_box)

        crop = image.crop(crop_box)
        save_pil_image(crop, accepted_path, image_format)
        metrics["saved"] = "written"
        metrics["created_output"] = True
        return True, None, metrics
