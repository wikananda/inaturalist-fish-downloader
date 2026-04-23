"""Image validation and accepted-image writing helpers."""

import argparse
import shutil
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - handled at runtime for optional validation.
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError


def pillow_available() -> bool:
    """Return whether Pillow imports required by validation are available."""
    return Image is not None and ImageOps is not None


def validate_image(path: Path, args: argparse.Namespace) -> tuple[bool, Optional[str], dict]:
    """Run phase-2 integrity and basic quality checks on a downloaded image."""
    metrics: dict[str, object] = {}

    if not path.exists():
        return False, "missing_file", metrics

    file_size = path.stat().st_size
    file_size_kb = file_size / 1024
    metrics["file_size_bytes"] = file_size
    metrics["file_size_kb"] = round(file_size_kb, 2)
    if args.min_file_size_kb > 0 and file_size_kb < args.min_file_size_kb:
        return False, "file_too_small", metrics

    if not pillow_available():
        return False, "pillow_not_installed", metrics

    try:
        with Image.open(path) as image:
            image_format = image.format
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
            metrics["width"] = width
            metrics["height"] = height
            metrics["format"] = image_format

            if args.min_width > 0 and width < args.min_width:
                return False, "width_too_small", metrics
            if args.min_height > 0 and height < args.min_height:
                return False, "height_too_small", metrics

            shorter_side = min(width, height)
            if shorter_side <= 0:
                return False, "invalid_dimensions", metrics

            aspect_ratio = max(width, height) / shorter_side
            metrics["aspect_ratio"] = round(aspect_ratio, 4)
            if args.max_aspect_ratio > 0 and aspect_ratio > args.max_aspect_ratio:
                return False, "aspect_ratio_too_extreme", metrics

            if args.min_intensity_range > 0:
                grayscale = image.convert("L")
                low, high = grayscale.getextrema()
                intensity_range = int(high) - int(low)
                metrics["intensity_range"] = intensity_range
                if intensity_range < args.min_intensity_range:
                    return False, "near_empty_image", metrics
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        metrics["error"] = str(exc)
        return False, "invalid_image", metrics

    return True, None, metrics


def save_accepted_image(raw_path: Path, accepted_path: Path, overwrite: bool) -> str:
    """Copy or normalize a valid raw image into the accepted image directory."""
    if accepted_path.exists() and not overwrite:
        return "accepted_existing"

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.resolve() == accepted_path.resolve():
        return "accepted"

    if not pillow_available():
        shutil.copy2(raw_path, accepted_path)
        return "accepted"

    with Image.open(raw_path) as image:
        image_format = image.format or "JPEG"
        image = ImageOps.exif_transpose(image)
        save_pil_image(image, accepted_path, image_format)
    return "accepted"


def save_pil_image(image, destination: Path, image_format: Optional[str]) -> None:
    """Save a Pillow image while handling JPEG mode/quality details."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_format = image_format or "JPEG"
    save_kwargs = {}
    if output_format.upper() in {"JPEG", "JPG"}:
        save_kwargs = {"quality": 95}
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
    image.save(destination, format=output_format, **save_kwargs)
