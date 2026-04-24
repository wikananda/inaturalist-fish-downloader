"""CLI parser, OmegaConf profile loading, and argument validation."""

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..common.utils import parse_csv_int_set, parse_csv_set
from .clip_filter import load_clip_prompts, validate_clip_import
from .detection import validate_detector_import
from .image_quality import pillow_available

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

FIELD_TO_PATH = {
    "species_file": "paths.species_file",
    "output_dir": "paths.output_dir",
    "raw_dir": "paths.raw_dir",
    "manifest_dir": "paths.manifest_dir",
    "images_per_species": "download.images_per_species",
    "candidate_multiplier": "download.candidate_multiplier",
    "max_candidates_per_species": "download.max_candidates_per_species",
    "quality_grade": "inat.quality_grade",
    "photo_size": "inat.photo_size",
    "place_id": "inat.place_id",
    "exclude_captive": "inat.exclude_captive",
    "alive_only": "inat.alive_only",
    "term_id": "inat.term_id",
    "term_value_id": "inat.term_value_id",
    "per_page": "inat.per_page",
    "max_pages": "inat.max_pages",
    "license_code": "inat.license_code",
    "sleep_seconds": "inat.sleep_seconds",
    "include_subspecies": "inat.include_subspecies",
    "overwrite": "download.overwrite",
    "redownload": "download.redownload",
    "retries": "inat.retries",
    "species_workers": "download.species_workers",
    "download_workers": "download.download_workers",
    "skip_image_validation": "validation.skip_image_validation",
    "min_width": "validation.min_width",
    "min_height": "validation.min_height",
    "min_file_size_kb": "validation.min_file_size_kb",
    "max_aspect_ratio": "validation.max_aspect_ratio",
    "min_intensity_range": "validation.min_intensity_range",
    "enable_detection": "detection.enable",
    "detector_weights": "detection.weights",
    "detector_device": "detection.device",
    "detector_confidence": "detection.confidence",
    "detector_imgsz": "detection.imgsz",
    "detector_class_names": "detection.class_names",
    "detector_class_ids": "detection.class_ids",
    "min_fish_area_ratio": "detection.min_fish_area_ratio",
    "crop_padding": "detection.crop_padding",
    "allow_multiple_fish": "detection.allow_multiple_fish",
    "enable_clip_filter": "clip.enable",
    "clip_model": "clip.model",
    "clip_cache_dir": "clip.cache_dir",
    "clip_device": "clip.device",
    "clip_threshold": "clip.threshold",
    "clip_prompts_file": "clip.prompts_file",
}

BOOL_FIELDS = {
    "exclude_captive",
    "alive_only",
    "include_subspecies",
    "overwrite",
    "skip_image_validation",
    "enable_detection",
    "allow_multiple_fish",
    "enable_clip_filter",
}

INT_FIELDS = {
    "images_per_species",
    "max_candidates_per_species",
    "place_id",
    "term_id",
    "per_page",
    "max_pages",
    "retries",
    "species_workers",
    "download_workers",
    "min_width",
    "min_height",
    "min_file_size_kb",
    "detector_imgsz",
}

FLOAT_FIELDS = {
    "candidate_multiplier",
    "sleep_seconds",
    "max_aspect_ratio",
    "min_intensity_range",
    "detector_confidence",
    "min_fish_area_ratio",
    "crop_padding",
    "clip_threshold",
}

CHOICE_FIELDS = {
    "quality_grade": ["any", "research", "needs_id", "casual"],
    "photo_size": ["square", "thumb", "small", "medium", "large", "original"],
}

HELP_TEXT = {
    "species_file": "Path to a text file containing one species per line.",
    "output_dir": "Directory to store accepted images after phase-2 validation.",
    "raw_dir": "Directory to store raw downloaded candidate images before validation.",
    "manifest_dir": "Directory to store candidates, accepted, rejected, and summary manifests.",
    "images_per_species": "Maximum number of accepted images to keep for each species.",
    "candidate_multiplier": "Collect this many candidate photos per final accepted target.",
    "max_candidates_per_species": "Hard cap for candidate photos scanned per species.",
    "quality_grade": "Observation quality filter. Use 'any' to skip this filter.",
    "photo_size": "Requested image size variant from iNaturalist photo URLs.",
    "place_id": "Optional iNaturalist place ID to use when fetching observation photos.",
    "exclude_captive": "Exclude observations marked captive/cultivated by passing captive=false.",
    "alive_only": "Require the iNaturalist Alive or Dead annotation to be Alive.",
    "term_id": "Optional iNaturalist annotation term_id filter.",
    "term_value_id": "Optional iNaturalist annotation term_value_id filter, e.g. '18' or '2,6'.",
    "per_page": "Observations to request per API page. Max is typically 200.",
    "max_pages": "Maximum observation pages to scan per species.",
    "license_code": "Optional photo license filter, for example 'cc-by' or 'cc-by-nc'.",
    "sleep_seconds": "Optional delay between image downloads.",
    "include_subspecies": "Allow non-exact taxon matches when resolving a species name.",
    "overwrite": "Redownload files even when the target image already exists.",
    "redownload": "Path to a file containing species to redownload. Overrides species_file and forces overwrite.",
    "retries": "Number of retries for failed HTTP requests.",
    "species_workers": "Number of species to process in parallel.",
    "download_workers": "Number of image downloads to run in parallel per species.",
    "skip_image_validation": "Skip phase-2 image integrity and dimension validation.",
    "min_width": "Minimum accepted image width in pixels after EXIF orientation.",
    "min_height": "Minimum accepted image height in pixels after EXIF orientation.",
    "min_file_size_kb": "Minimum accepted file size in KB.",
    "max_aspect_ratio": "Reject images whose longer side / shorter side exceeds this value.",
    "min_intensity_range": "Reject near-empty images whose grayscale max-min intensity range is below this value.",
    "enable_detection": "Run YOLO fish detection after image validation and save accepted crops.",
    "detector_weights": "Path to YOLO detector weights, for example models/fish-yolo.pt.",
    "detector_device": "Optional YOLO device, for example 'cpu', 'mps', or '0'.",
    "detector_confidence": "Minimum YOLO detection confidence.",
    "detector_imgsz": "YOLO inference image size.",
    "detector_class_names": "Optional comma-separated class names to accept as fish.",
    "detector_class_ids": "Optional comma-separated numeric class IDs to accept as fish.",
    "min_fish_area_ratio": "Reject detections whose box area / image area is below this value.",
    "crop_padding": "Padding around the selected fish bounding box as a fraction of box size.",
    "allow_multiple_fish": "Allow images with multiple fish detections; otherwise reject them for cleaner few-shot classes.",
    "enable_clip_filter": "Run CLIP context filtering after detection/cropping or accepted image preparation.",
    "clip_model": "CLIP model name or local path for Transformers.",
    "clip_cache_dir": "Directory used to cache downloaded CLIP model files.",
    "clip_device": "Optional CLIP device, for example 'cpu', 'mps', or 'cuda'.",
    "clip_threshold": "Minimum CLIP context score margin required to accept an image.",
    "clip_prompts_file": "Optional JSON file defining positive/negative CLIP prompts.",
}

OPTION_NAMES = {
    "license_code": "--license",
}


def _repo_config_candidates(value: str) -> list[Path]:
    raw = Path(value)
    candidates = [raw]
    if not raw.suffix:
        candidates.append(CONFIG_DIR / f"{value}.yaml")
        candidates.append(CONFIG_DIR / f"{value}.yml")
    candidates.append(CONFIG_DIR / raw)
    return candidates


def resolve_config_path(value: str) -> Path:
    """Resolve a user-provided config reference to a real YAML path."""
    for candidate in _repo_config_candidates(value):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Config file not found: {value}")


def _nested_get(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_default_config():
    """Load the bundled default downloader config."""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Default downloader config not found: {DEFAULT_CONFIG_PATH}")
    return OmegaConf.load(DEFAULT_CONFIG_PATH)


def flatten_config(cfg) -> dict[str, Any]:
    """Flatten the nested OmegaConf downloader schema into CLI field names."""
    container = OmegaConf.to_container(cfg, resolve=True)
    return {
        field: _nested_get(container, dotted_path)
        for field, dotted_path in FIELD_TO_PATH.items()
    }


def build_override_config(overrides: dict[str, Any]):
    """Convert flat CLI override values into nested OmegaConf structure."""
    data: dict[str, Any] = {}
    for field, value in overrides.items():
        dotted_path = FIELD_TO_PATH[field]
        current = data
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return OmegaConf.create(data)


def effective_config_yaml(cfg) -> str:
    """Render the merged downloader config as YAML."""
    return OmegaConf.to_yaml(cfg, resolve=True)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    """Build the downloader CLI parser with config-aware defaults in help text."""
    parser = argparse.ArgumentParser(
        description="Download iNaturalist observation photos for species listed in a text file."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Optional YAML config profile. Base defaults are always loaded from {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective merged config and exit.",
    )

    for field in FIELD_TO_PATH:
        option = OPTION_NAMES.get(field, f"--{field.replace('_', '-')}")
        help_text = HELP_TEXT[field]
        default_value = defaults.get(field)
        if default_value is not None:
            help_text = f"{help_text} Default: {default_value}"

        kwargs: dict[str, Any] = {
            "dest": field,
            "default": argparse.SUPPRESS,
            "help": help_text,
        }

        if field in BOOL_FIELDS:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif field in INT_FIELDS:
            kwargs["type"] = int
        elif field in FLOAT_FIELDS:
            kwargs["type"] = float
        if field in CHOICE_FIELDS:
            kwargs["choices"] = CHOICE_FIELDS[field]

        parser.add_argument(option, **kwargs)

    return parser


def parse_args() -> argparse.Namespace:
    """Parse downloader args from defaults + config file + CLI overrides."""
    default_cfg = load_default_config()
    parser = build_parser(flatten_config(default_cfg))
    raw_args = parser.parse_args()

    cli_overrides = vars(raw_args).copy()
    config_value = cli_overrides.pop("config", None)
    print_config = cli_overrides.pop("print_config", False)

    merged_cfg = default_cfg
    config_path = DEFAULT_CONFIG_PATH.resolve()
    if config_value:
        config_path = resolve_config_path(config_value)
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(config_path))
    if cli_overrides:
        merged_cfg = OmegaConf.merge(merged_cfg, build_override_config(cli_overrides))

    args = argparse.Namespace(**flatten_config(merged_cfg))
    args.config = str(config_value) if config_value else None
    args.config_path = str(config_path)
    args.effective_config = merged_cfg

    if print_config:
        print(effective_config_yaml(merged_cfg))
        raise SystemExit(0)

    return args


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
