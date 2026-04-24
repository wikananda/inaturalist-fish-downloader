"""CLI parser and argument validation for the downloader command."""

import argparse
import json
from pathlib import Path

from ..common.config import (
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SPECIES_FILE,
    DEFAULT_SPECIES_WORKERS,
    IMAGES_PER_SPECIES,
    VALID_GRADES,
)
from .clip_filter import load_clip_prompts, validate_clip_import
from .detection import validate_detector_import
from .image_quality import pillow_available
from ..common.utils import parse_csv_int_set, parse_csv_set


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the downloader workflow."""
    parser = argparse.ArgumentParser(
        description="Download iNaturalist observation photos for species listed in a text file."
    )
    parser.add_argument(
        "--species-file",
        default=DEFAULT_SPECIES_FILE,
        help=f"Path to a text file containing one species per line. Default: {DEFAULT_SPECIES_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to store accepted images after phase-2 validation. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help=(
            "Directory to store raw downloaded candidate images before validation. "
            f"Default: {DEFAULT_RAW_DIR}"
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        default=DEFAULT_MANIFEST_DIR,
        help=(
            "Directory to store candidates, accepted, rejected, and summary manifests. "
            f"Default: {DEFAULT_MANIFEST_DIR}"
        ),
    )
    parser.add_argument(
        "--images-per-species",
        type=int,
        default=IMAGES_PER_SPECIES,
        help=f"Maximum number of accepted images to keep for each species. Default: {IMAGES_PER_SPECIES}",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.0,
        help=(
            "Collect this many candidate photos per final accepted target. "
            "Example: 5 means inspect up to 5x --images-per-species. Default: 1"
        ),
    )
    parser.add_argument(
        "--max-candidates-per-species",
        type=int,
        default=None,
        help="Hard cap for candidate photos scanned per species. Default: no separate cap.",
    )
    parser.add_argument(
        "--quality-grade",
        default="research",
        choices=sorted(VALID_GRADES),
        help="Observation quality filter. Use 'any' to skip this filter. Default: research",
    )
    parser.add_argument(
        "--photo-size",
        default="original",
        choices=["square", "thumb", "small", "medium", "large", "original"],
        help="Requested image size variant from iNaturalist photo URLs. Default: original",
    )
    parser.add_argument(
        "--place-id",
        type=int,
        default=None,
        help="Optional iNaturalist place ID to use when fetching observation photos.",
    )
    parser.add_argument(
        "--exclude-captive",
        action="store_true",
        help="Exclude observations marked captive/cultivated by passing captive=false.",
    )
    parser.add_argument(
        "--alive-only",
        action="store_true",
        help=(
            "Require the iNaturalist Alive or Dead annotation to be Alive "
            "(term_id=17, term_value_id=18)."
        ),
    )
    parser.add_argument(
        "--term-id",
        type=int,
        default=None,
        help="Optional iNaturalist annotation term_id filter.",
    )
    parser.add_argument(
        "--term-value-id",
        default=None,
        help="Optional iNaturalist annotation term_value_id filter, e.g. '18' or '2,6'.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="Observations to request per API page. Max is typically 200. Default: 100",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum observation pages to scan per species. Default: 100",
    )
    parser.add_argument(
        "--license",
        dest="license_code",
        default=None,
        help="Optional photo license filter, for example 'cc-by' or 'cc-by-nc'.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between image downloads. Default: 0",
    )
    parser.add_argument(
        "--include-subspecies",
        action="store_true",
        help="Allow non-exact taxon matches when resolving a species name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even when the target image already exists.",
    )
    parser.add_argument(
        "--redownload",
        type=str,
        default=None,
        help="Path to a file containing species to redownload. Overrides --species-file and forces --overwrite.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of retries for failed HTTP requests. Default: 5",
    )
    parser.add_argument(
        "--species-workers",
        type=int,
        default=DEFAULT_SPECIES_WORKERS,
        help=f"Number of species to process in parallel. Default: {DEFAULT_SPECIES_WORKERS}",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help=f"Number of image downloads to run in parallel per species. Default: {DEFAULT_DOWNLOAD_WORKERS}",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip phase-2 image integrity and dimension validation.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=512,
        help="Minimum accepted image width in pixels after EXIF orientation. Default: 512",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=512,
        help="Minimum accepted image height in pixels after EXIF orientation. Default: 512",
    )
    parser.add_argument(
        "--min-file-size-kb",
        type=int,
        default=10,
        help="Minimum accepted file size in KB. Default: 10",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=4.0,
        help="Reject images whose longer side / shorter side exceeds this value. Default: 4.0",
    )
    parser.add_argument(
        "--min-intensity-range",
        type=int,
        default=10,
        help=(
            "Reject near-empty images whose grayscale max-min intensity range is below "
            "this value. Use 0 to disable. Default: 10"
        ),
    )
    parser.add_argument(
        "--enable-detection",
        action="store_true",
        help="Run YOLO fish detection after image validation and save accepted crops.",
    )
    parser.add_argument(
        "--detector-weights",
        default=None,
        help="Path to YOLO detector weights, for example models/fish-yolo.pt.",
    )
    parser.add_argument(
        "--detector-device",
        default=None,
        help="Optional YOLO device, for example 'cpu', 'mps', or '0'. Default: Ultralytics auto-select.",
    )
    parser.add_argument(
        "--detector-confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO detection confidence. Default: 0.25",
    )
    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=640,
        help="YOLO inference image size. Default: 640",
    )
    parser.add_argument(
        "--detector-class-names",
        default=None,
        help=(
            "Optional comma-separated class names to accept as fish. "
            "If omitted, all detector classes are accepted."
        ),
    )
    parser.add_argument(
        "--detector-class-ids",
        default=None,
        help=(
            "Optional comma-separated numeric class IDs to accept as fish. "
            "If omitted, all detector class IDs are accepted."
        ),
    )
    parser.add_argument(
        "--min-fish-area-ratio",
        type=float,
        default=0.03,
        help="Reject detections whose box area / image area is below this value. Default: 0.03",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.15,
        help="Padding around the selected fish bounding box as a fraction of box size. Default: 0.15",
    )
    parser.add_argument(
        "--allow-multiple-fish",
        action="store_true",
        help="Allow images with multiple fish detections; otherwise reject them for cleaner few-shot classes.",
    )
    parser.add_argument(
        "--enable-clip-filter",
        action="store_true",
        help="Run CLIP context filtering after detection/cropping or accepted image preparation.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name or local path for Transformers. Default: openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Optional CLIP device, for example 'cpu', 'mps', or 'cuda'. Default: auto-select.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.05,
        help="Minimum CLIP context score margin required to accept an image. Default: 0.05",
    )
    parser.add_argument(
        "--clip-prompts-file",
        default=None,
        help="Optional JSON file defining positive/negative CLIP prompts.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate downloader CLI arguments and populate derived detector filters."""
    if args.images_per_species <= 0:
        raise SystemExit("--images-per-species must be greater than 0")
    if args.candidate_multiplier < 1:
        raise SystemExit("--candidate-multiplier must be at least 1")
    if (
        args.max_candidates_per_species is not None
        and args.max_candidates_per_species < args.images_per_species
    ):
        raise SystemExit("--max-candidates-per-species must be >= --images-per-species")
    if args.per_page <= 0:
        raise SystemExit("--per-page must be greater than 0")
    if args.max_pages <= 0:
        raise SystemExit("--max-pages must be greater than 0")
    if args.species_workers <= 0:
        raise SystemExit("--species-workers must be greater than 0")
    if args.download_workers <= 0:
        raise SystemExit("--download-workers must be greater than 0")
    if args.term_value_id and args.term_id is None and not args.alive_only:
        raise SystemExit("--term-value-id requires --term-id unless --alive-only is used")
    if not args.skip_image_validation and not pillow_available():
        raise SystemExit(
            "Pillow is required for image validation. Install pillow or use --skip-image-validation."
        )
    if args.min_width < 0:
        raise SystemExit("--min-width must be greater than or equal to 0")
    if args.min_height < 0:
        raise SystemExit("--min-height must be greater than or equal to 0")
    if args.min_file_size_kb < 0:
        raise SystemExit("--min-file-size-kb must be greater than or equal to 0")
    if args.max_aspect_ratio < 0:
        raise SystemExit("--max-aspect-ratio must be greater than or equal to 0")
    if args.min_intensity_range < 0:
        raise SystemExit("--min-intensity-range must be greater than or equal to 0")

    if args.enable_detection:
        if not args.detector_weights:
            raise SystemExit("--enable-detection requires --detector-weights")
        if args.detector_confidence < 0 or args.detector_confidence > 1:
            raise SystemExit("--detector-confidence must be between 0 and 1")
        if args.detector_imgsz <= 0:
            raise SystemExit("--detector-imgsz must be greater than 0")
        if args.min_fish_area_ratio < 0 or args.min_fish_area_ratio > 1:
            raise SystemExit("--min-fish-area-ratio must be between 0 and 1")
        if args.crop_padding < 0:
            raise SystemExit("--crop-padding must be greater than or equal to 0")
        try:
            args.detector_class_id_set = parse_csv_int_set(args.detector_class_ids)
        except ValueError as exc:
            raise SystemExit("--detector-class-ids must be comma-separated integers") from exc
        args.detector_class_name_set = parse_csv_set(args.detector_class_names)
        try:
            validate_detector_import()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        args.detector_class_id_set = set()
        args.detector_class_name_set = set()

    if args.enable_clip_filter:
        try:
            validate_clip_import()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        try:
            (
                args.clip_positive_prompts,
                args.clip_negative_prompts,
            ) = load_clip_prompts(args.clip_prompts_file)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Invalid CLIP prompts configuration: {exc}") from exc
    else:
        args.clip_positive_prompts = []
        args.clip_negative_prompts = []


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve species, accepted image, raw image, and manifest paths."""
    species_file = Path(args.species_file)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    manifest_dir = Path(args.manifest_dir)
    return species_file, output_dir, raw_dir, manifest_dir
