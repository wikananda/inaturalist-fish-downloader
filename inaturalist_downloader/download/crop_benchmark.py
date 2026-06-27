"""Benchmark helpers for comparing fish cropper backends."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .detection import detection_class_allowed, get_detector_model
from .image_quality import Image, ImageOps, pillow_available, save_pil_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    """One raw image selected for cropper benchmarking."""

    image_id: str
    species: str
    raw_path: Path
    observation_id: Optional[str] = None
    photo_id: Optional[str] = None


@dataclass
class CropResult:
    """Benchmark result for one backend on one image."""

    backend: str
    success: bool
    reject_reason: Optional[str]
    inference_seconds: float
    crop_box_xyxy: Optional[tuple[int, int, int, int]] = None
    score: Optional[float] = None
    detection_count: int = 0
    mask_area_ratio: Optional[float] = None
    crop_area_ratio: Optional[float] = None
    crop_path: Optional[Path] = None
    extra: Optional[dict[str, Any]] = None


def load_benchmark_samples(
    *,
    manifest_path: Path,
    raw_dir: Path,
    max_images: int,
    max_per_species: int,
) -> list[ImageSample]:
    """Load a deterministic species-balanced sample from manifest or raw folders."""
    samples = _samples_from_manifest(manifest_path)
    if not samples:
        samples = _samples_from_raw_dir(raw_dir)
    return _balanced_sample(samples, max_images=max_images, max_per_species=max_per_species)


def _samples_from_manifest(manifest_path: Path) -> list[ImageSample]:
    if not manifest_path.exists():
        return []

    samples = []
    seen_paths = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_path_value = record.get("raw_path")
            if not raw_path_value:
                continue
            raw_path = Path(raw_path_value)
            if not raw_path.exists() or raw_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if raw_path in seen_paths:
                continue

            seen_paths.add(raw_path)
            species = str(
                record.get("canonical_name")
                or record.get("species_name")
                or raw_path.parent.name
            )
            samples.append(
                ImageSample(
                    image_id=raw_path.stem,
                    species=species,
                    raw_path=raw_path,
                    observation_id=_optional_str(record.get("observation_id")),
                    photo_id=_optional_str(record.get("photo_id")),
                )
            )

    return sorted(samples, key=lambda item: (item.species.casefold(), str(item.raw_path)))


def _samples_from_raw_dir(raw_dir: Path) -> list[ImageSample]:
    if not raw_dir.exists():
        return []

    samples = []
    for raw_path in sorted(raw_dir.rglob("*")):
        if not raw_path.is_file() or raw_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        samples.append(
            ImageSample(
                image_id=raw_path.stem,
                species=raw_path.parent.name,
                raw_path=raw_path,
            )
        )
    return samples


def _balanced_sample(
    samples: Iterable[ImageSample],
    *,
    max_images: int,
    max_per_species: int,
) -> list[ImageSample]:
    grouped: dict[str, list[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.species, []).append(sample)

    selected = []
    for species in sorted(grouped):
        species_samples = grouped[species][:max_per_species]
        for sample in species_samples:
            if len(selected) >= max_images:
                return selected
            selected.append(sample)
    return selected


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def box_area_ratio(box: tuple[int, int, int, int], width: int, height: int) -> float:
    """Return box area / image area."""
    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    return box_area / max(width * height, 1)


def box_iou(
    a: Optional[tuple[int, int, int, int]],
    b: Optional[tuple[int, int, int, int]],
) -> Optional[float]:
    """Return intersection-over-union for two xyxy boxes."""
    if a is None or b is None:
        return None

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return None
    return intersection / union


def padded_crop_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Clip a padded xyxy crop box to image bounds."""
    x1, y1, x2, y2 = box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    pad_x = box_width * padding
    pad_y = box_height * padding
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(width, int(math.ceil(x2 + pad_x))),
        min(height, int(math.ceil(y2 + pad_y))),
    )


def run_yolo_cropper(
    *,
    image_path: Path,
    output_path: Path,
    weights: str,
    device: Optional[str],
    confidence: float,
    imgsz: int,
    class_ids: set[int],
    class_names: set[str],
    min_area_ratio: float,
    crop_padding: float,
) -> CropResult:
    """Run the existing YOLO detector and save its selected crop."""
    if not pillow_available():
        return CropResult("yolo", False, "pillow_not_installed", 0.0)

    started = time.perf_counter()
    model = get_detector_model(weights)
    with Image.open(image_path) as source_image:
        image_format = source_image.format
        image = ImageOps.exif_transpose(source_image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        width, height = image.size
        image_area = max(width * height, 1)

        predict_kwargs = {
            "source": image,
            "conf": confidence,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device and device != "auto":
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)

        detections = []
        result = results[0]
        if result.boxes is not None:
            xyxy_values = result.boxes.xyxy.cpu().tolist()
            conf_values = result.boxes.conf.cpu().tolist()
            cls_values = result.boxes.cls.cpu().tolist()
            names = result.names or {}
            for xyxy, score, class_value in zip(xyxy_values, conf_values, cls_values):
                class_id = int(class_value)
                class_name = str(names.get(class_id, class_id))
                if not detection_class_allowed(class_id, class_name, class_ids, class_names):
                    continue
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "score": float(score),
                        "area_ratio": area_ratio,
                        "selection_score": float(score) * area_ratio,
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

        inference_seconds = time.perf_counter() - started
        if not detections:
            return CropResult("yolo", False, "no_fish_detected", inference_seconds)

        selected = max(detections, key=lambda item: item["selection_score"])
        if selected["area_ratio"] < min_area_ratio:
            return CropResult(
                "yolo",
                False,
                "fish_too_small",
                inference_seconds,
                score=selected["score"],
                detection_count=len(detections),
                crop_area_ratio=selected["area_ratio"],
            )

        crop_box = padded_crop_box(
            selected["box"], width=width, height=height, padding=crop_padding
        )
        crop = image.crop(crop_box)
        save_pil_image(crop, output_path, image_format)
        return CropResult(
            "yolo",
            True,
            None,
            inference_seconds,
            crop_box_xyxy=crop_box,
            score=selected["score"],
            detection_count=len(detections),
            crop_area_ratio=box_area_ratio(crop_box, width, height),
            crop_path=output_path,
            extra={
                "class_id": selected["class_id"],
                "class_name": selected["class_name"],
            },
        )


def run_sam3_cropper(
    *,
    image_path: Path,
    output_path: Path,
    prompt: str,
    device: Optional[str],
    crop_padding: float,
) -> CropResult:
    """Run SAM 3 text-prompt segmentation and save the selected mask box crop."""
    if not pillow_available():
        return CropResult("sam3", False, "pillow_not_installed", 0.0)

    started = time.perf_counter()
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        return CropResult(
            "sam3",
            False,
            "sam3_not_available",
            time.perf_counter() - started,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )

    try:
        model = _get_sam3_model(build_sam3_image_model, device)
        processor = Sam3Processor(model)
        with Image.open(image_path) as source_image:
            image_format = source_image.format
            image = ImageOps.exif_transpose(source_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            width, height = image.size
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get("masks")
            boxes = output.get("boxes")
            scores = output.get("scores")
            selected = _select_sam3_instance(masks=masks, boxes=boxes, scores=scores)
            inference_seconds = time.perf_counter() - started
            if selected is None:
                return CropResult("sam3", False, "no_fish_detected", inference_seconds)

            box, score, mask_area_ratio = selected
            crop_box = padded_crop_box(box, width=width, height=height, padding=crop_padding)
            crop = image.crop(crop_box)
            save_pil_image(crop, output_path, image_format)
            return CropResult(
                "sam3",
                True,
                None,
                inference_seconds,
                crop_box_xyxy=crop_box,
                score=score,
                detection_count=_safe_len(boxes),
                mask_area_ratio=mask_area_ratio,
                crop_area_ratio=box_area_ratio(crop_box, width, height),
                crop_path=output_path,
            )
    except Exception as exc:
        return CropResult(
            "sam3",
            False,
            "sam3_error",
            time.perf_counter() - started,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )


_SAM3_MODEL = None
_SAM3_MODEL_DEVICE = None


def _get_sam3_model(build_sam3_image_model, device: Optional[str]):
    global _SAM3_MODEL, _SAM3_MODEL_DEVICE
    target_device = _resolve_device(device)
    if _SAM3_MODEL is not None and _SAM3_MODEL_DEVICE == target_device:
        return _SAM3_MODEL
    model = build_sam3_image_model()
    if target_device and hasattr(model, "to"):
        model = model.to(target_device)
    if hasattr(model, "eval"):
        model.eval()
    _SAM3_MODEL = model
    _SAM3_MODEL_DEVICE = target_device
    return model


def _resolve_device(device: Optional[str]) -> Optional[str]:
    if not device or device == "auto":
        try:
            import torch
        except Exception:
            return None
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _select_sam3_instance(*, masks, boxes, scores):
    if boxes is None or scores is None or _safe_len(boxes) == 0:
        return None
    box_values = _tensor_to_list(boxes)
    score_values = _tensor_to_list(scores)
    mask_values = _tensor_to_list(masks) if masks is not None else None
    best_index = max(range(len(box_values)), key=lambda index: float(score_values[index]))
    box = tuple(float(value) for value in box_values[best_index][:4])
    score = float(score_values[best_index])
    mask_area_ratio = None
    if mask_values is not None:
        flat_mask = _flatten(mask_values[best_index])
        if flat_mask:
            mask_area_ratio = sum(1 for value in flat_mask if float(value) > 0) / len(flat_mask)
    return box, score, mask_area_ratio


def _safe_len(value) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return int(value.shape[0]) if hasattr(value, "shape") else 0


def _tensor_to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flatten(values):
    if not isinstance(values, list):
        return [values]
    flattened = []
    for value in values:
        flattened.extend(_flatten(value))
    return flattened


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write benchmark row metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "species",
        "raw_path",
        "backend",
        "success",
        "reject_reason",
        "inference_seconds",
        "detection_count",
        "score",
        "crop_area_ratio",
        "mask_area_ratio",
        "crop_box_xyxy",
        "box_iou_with_other",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate benchmark metrics."""
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        backend = str(row["backend"])
        item = summary.setdefault(
            backend,
            {
                "images": 0,
                "successes": 0,
                "failures": 0,
                "total_inference_seconds": 0.0,
                "reject_reasons": {},
            },
        )
        item["images"] += 1
        item["total_inference_seconds"] += float(row.get("inference_seconds") or 0)
        if row.get("success"):
            item["successes"] += 1
        else:
            item["failures"] += 1
            reason = row.get("reject_reason") or "unknown"
            item["reject_reasons"][reason] = item["reject_reasons"].get(reason, 0) + 1

    for item in summary.values():
        images = max(item["images"], 1)
        item["success_rate"] = item["successes"] / images
        item["mean_inference_seconds"] = item["total_inference_seconds"] / images

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def make_contact_sheet(
    *,
    raw_path: Path,
    crop_paths: dict[str, Optional[Path]],
    output_path: Path,
    tile_size: tuple[int, int] = (256, 256),
) -> None:
    """Create a small raw/crop contact sheet for manual inspection."""
    if not pillow_available():
        return

    labels = ["raw", *crop_paths.keys()]
    tiles = []
    for label in labels:
        path = raw_path if label == "raw" else crop_paths.get(label)
        if path is None or not path.exists():
            tile = Image.new("RGB", tile_size, color=(240, 240, 240))
        else:
            with Image.open(path) as image:
                tile = ImageOps.exif_transpose(image).convert("RGB")
                tile.thumbnail(tile_size)
                canvas = Image.new("RGB", tile_size, color=(255, 255, 255))
                x = (tile_size[0] - tile.width) // 2
                y = (tile_size[1] - tile.height) // 2
                canvas.paste(tile, (x, y))
                tile = canvas
        tiles.append(tile)

    sheet = Image.new("RGB", (tile_size[0] * len(tiles), tile_size[1]), color="white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile_size[0], 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
