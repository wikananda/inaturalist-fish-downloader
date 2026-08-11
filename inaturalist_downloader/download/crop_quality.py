"""Deterministic quality gates for fish crops and detector consistency.

The semantic filter is deliberately not responsible for crop geometry.  This
module turns measurable failure modes (tiny fish, edge truncation, background-
dominated crops, low information, and inconsistent re-detection) into explicit
manifest metrics and rejection reasons.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from .image_quality import ImageFilter, ImageOps, ImageStat, pillow_available


def _arg(args, name: str, default):
    return getattr(args, name, default)


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_edge_margins(
    box: Sequence[float], width: int, height: int
) -> dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    width = max(int(width), 1)
    height = max(int(height), 1)
    return {
        "left": max(0.0, x1) / width,
        "top": max(0.0, y1) / height,
        "right": max(0.0, width - x2) / width,
        "bottom": max(0.0, height - y2) / height,
    }


def crop_geometry_metrics(
    *,
    source_size: tuple[int, int],
    fish_box: Sequence[float],
    crop_box: Sequence[float],
) -> dict[str, Any]:
    """Measure fish size, fill, and edge margins in source and crop space."""
    source_width, source_height = source_size
    fx1, fy1, fx2, fy2 = [float(value) for value in fish_box[:4]]
    cx1, cy1, cx2, cy2 = [float(value) for value in crop_box[:4]]
    crop_width = max(0.0, cx2 - cx1)
    crop_height = max(0.0, cy2 - cy1)
    fish_width = max(0.0, fx2 - fx1)
    fish_height = max(0.0, fy2 - fy1)
    source_area = max(float(source_width * source_height), 1.0)
    crop_area = max(crop_width * crop_height, 1.0)
    fish_area = fish_width * fish_height
    crop_space_box = (fx1 - cx1, fy1 - cy1, fx2 - cx1, fy2 - cy1)
    source_margins = _normalized_edge_margins(
        fish_box, source_width, source_height
    )
    crop_margins = _normalized_edge_margins(
        crop_space_box, round(crop_width), round(crop_height)
    )
    return {
        "source_width": int(source_width),
        "source_height": int(source_height),
        "crop_width": int(round(crop_width)),
        "crop_height": int(round(crop_height)),
        "crop_short_side": int(round(min(crop_width, crop_height))),
        "crop_long_side": int(round(max(crop_width, crop_height))),
        "fish_bbox_width": round(fish_width, 3),
        "fish_bbox_height": round(fish_height, 3),
        "fish_source_area_ratio": round(fish_area / source_area, 6),
        "fish_crop_area_ratio": round(fish_area / crop_area, 6),
        "source_edge_margins": {
            key: round(value, 6) for key, value in source_margins.items()
        },
        "source_min_edge_margin_ratio": round(min(source_margins.values()), 6),
        "crop_edge_margins": {
            key: round(value, 6) for key, value in crop_margins.items()
        },
        "crop_min_edge_margin_ratio": round(min(crop_margins.values()), 6),
        "fish_box_in_crop_xyxy": [round(value, 3) for value in crop_space_box],
    }


def crop_visual_metrics(crop) -> dict[str, Any]:
    """Return inexpensive no-reference sharpness and information metrics."""
    if not pillow_available() or ImageFilter is None or ImageStat is None:
        return {"available": False}

    image = ImageOps.exif_transpose(crop)
    grayscale = image.convert("L")
    sample = grayscale.copy()
    sample.thumbnail((256, 256))
    low, high = sample.getextrema()
    entropy = float(sample.entropy())
    edges = sample.filter(ImageFilter.FIND_EDGES)
    width, height = edges.size
    if width > 2 and height > 2:
        edges = edges.crop((1, 1, width - 1, height - 1))
    edge_variance = float(ImageStat.Stat(edges).var[0])
    return {
        "available": True,
        "sample_width": sample.width,
        "sample_height": sample.height,
        "intensity_range": int(high) - int(low),
        "entropy": round(entropy, 6),
        "edge_variance": round(edge_variance, 6),
    }


def evaluate_crop_quality(
    *,
    crop,
    source_size: tuple[int, int],
    fish_box: Sequence[float],
    crop_box: Sequence[float],
    args,
    mask_area_ratio: Optional[float] = None,
) -> tuple[bool, Optional[str], dict[str, Any]]:
    """Apply deterministic final-crop checks and return manifest-ready metrics."""
    geometry = crop_geometry_metrics(
        source_size=source_size,
        fish_box=fish_box,
        crop_box=crop_box,
    )
    visual = crop_visual_metrics(crop)
    metrics: dict[str, Any] = {
        "enabled": bool(_arg(args, "enable_crop_quality", False)),
        "geometry": geometry,
        "visual": visual,
    }
    if mask_area_ratio is not None:
        source_area = max(source_size[0] * source_size[1], 1)
        crop_area = max(geometry["crop_width"] * geometry["crop_height"], 1)
        mask_crop_area_ratio = float(mask_area_ratio) * source_area / crop_area
        metrics["mask_source_area_ratio"] = round(float(mask_area_ratio), 6)
        metrics["mask_crop_area_ratio"] = round(mask_crop_area_ratio, 6)

    if not metrics["enabled"]:
        return True, None, metrics

    checks = (
        (
            geometry["crop_short_side"]
            < int(_arg(args, "crop_min_short_side", 0)),
            "crop_short_side_too_small",
        ),
        (
            geometry["crop_long_side"]
            < int(_arg(args, "crop_min_long_side", 0)),
            "crop_long_side_too_small",
        ),
        (
            geometry["fish_bbox_width"]
            < float(_arg(args, "min_fish_bbox_width", 0)),
            "fish_bbox_width_too_small",
        ),
        (
            geometry["fish_bbox_height"]
            < float(_arg(args, "min_fish_bbox_height", 0)),
            "fish_bbox_height_too_small",
        ),
        (
            geometry["fish_crop_area_ratio"]
            < float(_arg(args, "min_fish_crop_area_ratio", 0)),
            "fish_too_small_in_crop",
        ),
        (
            float(_arg(args, "max_fish_crop_area_ratio", 1)) > 0
            and geometry["fish_crop_area_ratio"]
            > float(_arg(args, "max_fish_crop_area_ratio", 1)),
            "fish_overfills_crop",
        ),
        (
            geometry["source_min_edge_margin_ratio"]
            < float(_arg(args, "min_source_edge_margin_ratio", 0)),
            "fish_touches_source_edge",
        ),
        (
            geometry["crop_min_edge_margin_ratio"]
            < float(_arg(args, "min_crop_edge_margin_ratio", 0)),
            "fish_touches_crop_edge",
        ),
    )
    for failed, reason in checks:
        if failed:
            metrics["passed"] = False
            metrics["reject_reason"] = reason
            return False, reason, metrics

    if mask_area_ratio is not None:
        mask_crop_ratio = float(metrics["mask_crop_area_ratio"])
        if mask_crop_ratio < float(_arg(args, "sam_min_mask_crop_area_ratio", 0)):
            reason = "sam_mask_too_small_in_crop"
            metrics.update(passed=False, reject_reason=reason)
            return False, reason, metrics
        maximum = float(_arg(args, "sam_max_mask_crop_area_ratio", 1))
        if maximum > 0 and mask_crop_ratio > maximum:
            reason = "sam_mask_overfills_crop"
            metrics.update(passed=False, reject_reason=reason)
            return False, reason, metrics

    if visual.get("available"):
        if visual["edge_variance"] < float(
            _arg(args, "min_crop_edge_variance", 0)
        ):
            reason = "crop_too_blurry_or_flat"
            metrics.update(passed=False, reject_reason=reason)
            return False, reason, metrics
        if visual["entropy"] < float(_arg(args, "min_crop_entropy", 0)):
            reason = "crop_information_too_low"
            metrics.update(passed=False, reject_reason=reason)
            return False, reason, metrics

    metrics["passed"] = True
    metrics["reject_reason"] = None
    return True, None, metrics


def evaluate_redetection_quality(
    *,
    boxes: Iterable[dict[str, Any]],
    expected_box: Sequence[float],
    crop_size: tuple[int, int],
    args,
) -> tuple[bool, Optional[str], dict[str, Any]]:
    """Validate a second YOLO pass on the saved crop."""
    minimum_confidence = float(_arg(args, "crop_redetect_confidence", 0))
    candidates = [
        dict(box)
        for box in boxes
        if float(box.get("confidence", 0)) >= minimum_confidence
    ]
    metrics: dict[str, Any] = {
        "enabled": bool(_arg(args, "crop_redetect", False)),
        "confidence_threshold": minimum_confidence,
        "detection_count": len(candidates),
        "detections": candidates,
    }
    if not metrics["enabled"]:
        return True, None, metrics
    if not candidates:
        reason = "crop_redetect_no_fish"
        metrics.update(passed=False, reject_reason=reason)
        return False, reason, metrics
    if bool(_arg(args, "crop_redetect_require_single", True)) and len(candidates) != 1:
        reason = "crop_redetect_multiple_fish"
        metrics.update(passed=False, reject_reason=reason)
        return False, reason, metrics

    selected = max(
        candidates,
        key=lambda item: (
            float(item.get("confidence", 0))
            * float(item.get("area_ratio", 0))
        ),
    )
    selected_box = selected["bbox_xyxy"]
    agreement_iou = box_iou(expected_box, selected_box)
    fill_ratio = _box_area(selected_box) / max(crop_size[0] * crop_size[1], 1)
    margins = _normalized_edge_margins(selected_box, *crop_size)
    metrics.update(
        {
            "selected_detection": selected,
            "expected_box_xyxy": [round(float(value), 3) for value in expected_box],
            "agreement_iou": round(agreement_iou, 6),
            "fish_crop_area_ratio": round(fill_ratio, 6),
            "min_edge_margin_ratio": round(min(margins.values()), 6),
        }
    )

    checks = (
        (
            fill_ratio < float(_arg(args, "crop_redetect_min_area_ratio", 0)),
            "crop_redetect_fish_too_small",
        ),
        (
            float(_arg(args, "crop_redetect_max_area_ratio", 1)) > 0
            and fill_ratio
            > float(_arg(args, "crop_redetect_max_area_ratio", 1)),
            "crop_redetect_fish_overfills_crop",
        ),
        (
            agreement_iou < float(_arg(args, "crop_redetect_min_iou", 0)),
            "crop_redetect_box_disagreement",
        ),
        (
            min(margins.values())
            < float(_arg(args, "crop_redetect_min_edge_margin_ratio", 0)),
            "crop_redetect_fish_touches_edge",
        ),
    )
    for failed, reason in checks:
        if failed:
            metrics.update(passed=False, reject_reason=reason)
            return False, reason, metrics

    metrics.update(passed=True, reject_reason=None)
    return True, None, metrics
