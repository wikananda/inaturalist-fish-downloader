"""Download and validate iNaturalist fish images for classification datasets.

This module powers the `inat-download` CLI and the backward-compatible
`fish_downloader.py` wrapper. The current implementation covers phase 1 and
phase 2 of the improvement plan: iNaturalist candidate acquisition,
raw/accepted directory separation, manifest writing, and basic image
validation.
"""

import argparse
import concurrent.futures
import csv
import json
import math
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from http.client import IncompleteRead

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - handled at runtime for optional validation.
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError


API_BASE = "https://api.inaturalist.org/v1"
DEFAULT_SPECIES_FILE = "species.txt"
DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_RAW_DIR = "downloads_raw"
DEFAULT_MANIFEST_DIR = "manifests"
DEFAULT_TIMEOUT = 30
IMAGES_PER_SPECIES = 30
DEFAULT_SPECIES_WORKERS = 5
DEFAULT_DOWNLOAD_WORKERS = 10
USER_AGENT = "inaturalist-downloader/1.0"
VALID_GRADES = {"any", "research", "needs_id", "casual"}
PRINT_LOCK = threading.Lock()
MANIFEST_LOCK = threading.Lock()
DETECTOR_LOCK = threading.Lock()
DETECTOR_MODEL = None
DETECTOR_MODEL_PATH = None
ALIVE_OR_DEAD_TERM_ID = 17
ALIVE_TERM_VALUE_ID = 18


def safe_print(message: str) -> None:
    """Print a message without interleaving output from worker threads.

    Args:
        message: Text to print to standard output.
    """
    with PRINT_LOCK:
        print(message)


def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSON Lines manifest in a thread-safe way.

    Args:
        path: Manifest file to append to.
        records: Serializable dictionaries to write as one JSON object per line.
    """
    if not records:
        return

    with MANIFEST_LOCK:
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_species_summary(path: Path, row: dict) -> None:
    """Append one species-level summary row to a TSV file.

    The header is written automatically when the file does not yet exist.

    Args:
        path: TSV summary file path.
        row: Species summary values keyed by field name.
    """
    fieldnames = [
        "run_id",
        "species_name",
        "canonical_name",
        "taxon_id",
        "candidates",
        "downloaded",
        "download_failed",
        "accepted",
        "rejected",
        "unused_valid",
    ]

    with MANIFEST_LOCK:
        should_write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if should_write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the downloader workflow.

    Returns:
        Parsed CLI options controlling species input, iNaturalist filters,
        candidate oversampling, output directories, concurrency, retries, and
        phase-2 image validation thresholds.
    """
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
    return parser.parse_args()


def load_species(path: Path) -> list[str]:
    """Load species names from a newline-delimited text file.

    Blank lines and comment lines beginning with `#` are ignored.

    Args:
        path: Path to `species.txt` or a redownload list.

    Returns:
        Species names in file order.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Species file not found: {path}")

    species = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        species.append(line)
    return species


def slugify(value: str) -> str:
    """Convert a species/taxon name into a filesystem-safe folder slug.

    Args:
        value: Raw name, usually a scientific species name.

    Returns:
        Lowercase ASCII-ish slug with non-alphanumeric runs replaced by `_`.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "species"


def http_get_bytes(url: str, params: Optional[dict] = None, retries: int = 5) -> bytes:
    """Fetch bytes from a URL with retry/backoff.

    Args:
        url: Base URL to request.
        params: Optional query string parameters.
        retries: Number of attempts before failing.

    Returns:
        Response body bytes.

    Raises:
        RuntimeError: If all attempts fail.
    """
    final_url = url
    if params:
        final_url = f"{url}?{urlencode(params)}"

    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(final_url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Request failed for {final_url}: {last_error}")


def api_get(path: str, retries: int = 5, **params) -> dict:
    """Fetch and decode JSON from an iNaturalist API path.

    Args:
        path: API path relative to `API_BASE`, such as `/observations`.
        retries: Number of HTTP attempts before failing.
        **params: Query string parameters passed to the API.

    Returns:
        Decoded JSON response.
    """
    payload = http_get_bytes(f"{API_BASE}{path}", params=params, retries=retries)
    return json.loads(payload.decode("utf-8"))


def resolve_taxon_id(species_name: str, include_subspecies: bool, retries: int = 5) -> tuple[int, str]:
    """Resolve a species name to an iNaturalist taxon ID and canonical name.

    Args:
        species_name: Species name read from the species file.
        include_subspecies: Allow partial or lower-rank matches if no exact
            species-level name is found.
        retries: Number of API attempts before failing.

    Returns:
        Tuple of `(taxon_id, canonical_name)`.

    Raises:
        ValueError: If no suitable taxon match is found.
    """
    payload = api_get("/taxa/autocomplete", q=species_name, per_page=30, retries=retries)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No taxon found for '{species_name}'")

    normalized = species_name.casefold()
    exact = []
    partial = []

    for item in results:
        candidate_names = {
            str(item.get("name", "")).casefold(),
            str(item.get("matched_term", "")).casefold(),
            str(item.get("preferred_common_name", "")).casefold(),
        }
        if normalized in candidate_names:
            exact.append(item)
        elif normalized in str(item.get("name", "")).casefold():
            partial.append(item)

    chosen = None
    if exact:
        chosen = exact[0]
    elif include_subspecies and partial:
        chosen = partial[0]
    elif include_subspecies and results:
        chosen = results[0]

    if chosen is None:
        raise ValueError(
            f"No exact taxon match for '{species_name}'. Try --include-subspecies if needed."
        )

    return int(chosen["id"]), str(chosen.get("name") or species_name)


def iter_observation_photos(
    taxon_id: int,
    quality_grade: str,
    per_page: int,
    max_pages: int,
    license_code: Optional[str],
    place_id: Optional[int],
    exclude_captive: bool,
    term_id: Optional[int],
    term_value_id: Optional[str],
    retries: int = 5,
) -> Iterator[dict]:
    """Yield photo metadata from matching iNaturalist observations.

    The query is paginated and constrained to observations with photos. Optional
    filters include quality grade, license, place, captive status, and
    annotation term/value IDs.

    Args:
        taxon_id: iNaturalist taxon ID to search.
        quality_grade: Observation quality grade, or `any` to omit the filter.
        per_page: Observation page size.
        max_pages: Maximum number of observation pages to scan.
        license_code: Optional photo license code.
        place_id: Optional iNaturalist place ID.
        exclude_captive: Pass `captive=false` when true.
        term_id: Optional iNaturalist annotation term ID.
        term_value_id: Optional annotation term value ID or comma-separated IDs.
        retries: Number of API attempts before failing.

    Yields:
        Dictionaries containing observation, photo, license, user, and date
        metadata needed for candidate manifests.
    """
    for page in range(1, max_pages + 1):
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "page": page,
            "per_page": per_page,
            "order_by": "votes",
            "order": "desc",
        }
        if quality_grade != "any":
            params["quality_grade"] = quality_grade
        if license_code:
            params["photo_license"] = license_code
        if place_id is not None:
            params["place_id"] = place_id
        if exclude_captive:
            params["captive"] = "false"
        if term_id is not None:
            params["term_id"] = term_id
        if term_value_id:
            params["term_value_id"] = term_value_id

        payload = api_get("/observations", retries=retries, **params)
        results = payload.get("results", [])
        if not results:
            return

        for observation in results:
            user = observation.get("user") or {}
            for photo in observation.get("photos", []):
                yield {
                    "observation_id": observation.get("id"),
                    "photo_id": photo.get("id"),
                    "url": photo.get("url"),
                    "license_code": photo.get("license_code"),
                    "quality_grade": observation.get("quality_grade"),
                    "observed_on": observation.get("observed_on"),
                    "time_observed_at": observation.get("time_observed_at"),
                    "captive": observation.get("captive"),
                    "place_guess": observation.get("place_guess"),
                    "user_id": user.get("id"),
                    "user_login": user.get("login"),
                }


def photo_url_for_size(url: str, size: str) -> str:
    """Rewrite an iNaturalist photo URL to request a target size variant.

    Args:
        url: Original iNaturalist photo URL.
        size: One of `square`, `thumb`, `small`, `medium`, `large`, or
            `original`.

    Returns:
        URL with the final size segment replaced when it matches the expected
        iNaturalist pattern.
    """
    pattern = r"/(square|thumb|small|medium|large|original)\.(jpg|jpeg|png)$"
    return re.sub(pattern, rf"/{size}.\2", url, flags=re.IGNORECASE)


def infer_extension(url: str) -> str:
    """Infer an image file extension from a URL.

    Args:
        url: Image URL.

    Returns:
        `.jpg`, `.jpeg`, or `.png` when present in the URL, otherwise `.jpg`.
    """
    match = re.search(r"\.(jpg|jpeg|png)(?:\?|$)", url, flags=re.IGNORECASE)
    return f".{match.group(1).lower()}" if match else ".jpg"


def effective_annotation_filter(args: argparse.Namespace) -> tuple[Optional[int], Optional[str]]:
    """Resolve CLI annotation settings into API term filters.

    `--alive-only` takes precedence and maps to iNaturalist's Alive annotation
    constants used by this script. Otherwise explicit `--term-id` and
    `--term-value-id` values are returned.

    Args:
        args: Parsed CLI options.

    Returns:
        Tuple of `(term_id, term_value_id)` where either value can be `None`.
    """
    if args.alive_only:
        return ALIVE_OR_DEAD_TERM_ID, str(ALIVE_TERM_VALUE_ID)
    return args.term_id, args.term_value_id


def candidate_limit_for_args(args: argparse.Namespace) -> int:
    """Calculate how many candidate photos to collect for one species.

    The limit is at least `--images-per-species`, scaled by
    `--candidate-multiplier`, and optionally capped by
    `--max-candidates-per-species`.

    Args:
        args: Parsed CLI options.

    Returns:
        Candidate photo limit for one species.
    """
    candidate_limit = max(
        args.images_per_species,
        math.ceil(args.images_per_species * args.candidate_multiplier),
    )
    if args.max_candidates_per_species is not None:
        candidate_limit = min(candidate_limit, args.max_candidates_per_species)
    return candidate_limit


def parse_csv_set(value: Optional[str]) -> set[str]:
    """Parse a comma-separated CLI value into a normalized string set.

    Args:
        value: Comma-separated text or `None`.

    Returns:
        Case-folded, stripped values. Empty items are ignored.
    """
    if not value:
        return set()
    return {item.strip().casefold() for item in value.split(",") if item.strip()}


def parse_csv_int_set(value: Optional[str]) -> set[int]:
    """Parse comma-separated integer IDs from a CLI value.

    Args:
        value: Comma-separated integers or `None`.

    Returns:
        Parsed integer set.

    Raises:
        ValueError: If any item is not an integer.
    """
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def validate_image(path: Path, args: argparse.Namespace) -> tuple[bool, Optional[str], dict]:
    """Run phase-2 integrity and basic quality checks on a downloaded image.

    Checks include file existence, file size, Pillow decodeability, EXIF-aware
    dimensions, minimum width/height, maximum aspect ratio, and near-empty
    grayscale intensity range.

    Args:
        path: Raw downloaded image path.
        args: Parsed CLI options containing validation thresholds.

    Returns:
        Tuple of `(is_valid, reject_reason, metrics)`. `reject_reason` is
        `None` when the image is valid. `metrics` contains measured properties
        such as file size, width, height, format, aspect ratio, and intensity
        range when available.
    """
    metrics: dict[str, object] = {}

    if not path.exists():
        return False, "missing_file", metrics

    file_size = path.stat().st_size
    file_size_kb = file_size / 1024
    metrics["file_size_bytes"] = file_size
    metrics["file_size_kb"] = round(file_size_kb, 2)
    if args.min_file_size_kb > 0 and file_size_kb < args.min_file_size_kb:
        return False, "file_too_small", metrics

    if Image is None or ImageOps is None:
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
    """Copy or normalize a valid raw image into the accepted image directory.

    When Pillow is available, EXIF orientation is applied before saving. JPEGs
    are saved with high quality. Existing accepted files are preserved unless
    `overwrite` is true.

    Args:
        raw_path: Raw candidate image path.
        accepted_path: Destination path under the accepted output directory.
        overwrite: Replace an existing accepted file when true.

    Returns:
        Status string: `accepted` or `accepted_existing`.
    """
    if accepted_path.exists() and not overwrite:
        return "accepted_existing"

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.resolve() == accepted_path.resolve():
        return "accepted"

    if Image is None or ImageOps is None:
        shutil.copy2(raw_path, accepted_path)
        return "accepted"

    with Image.open(raw_path) as image:
        image_format = image.format or "JPEG"
        image = ImageOps.exif_transpose(image)
        save_kwargs = {}
        if image_format.upper() in {"JPEG", "JPG"}:
            save_kwargs = {"quality": 95}
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
        image.save(accepted_path, format=image_format, **save_kwargs)
    return "accepted"


def save_pil_image(image, destination: Path, image_format: Optional[str]) -> None:
    """Save a Pillow image while handling JPEG mode/quality details.

    Args:
        image: Pillow image object to save.
        destination: Output path.
        image_format: Preferred format from the source image, or `None`.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_format = image_format or "JPEG"
    save_kwargs = {}
    if output_format.upper() in {"JPEG", "JPG"}:
        save_kwargs = {"quality": 95}
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
    image.save(destination, format=output_format, **save_kwargs)


def get_detector_model(weights_path: str):
    """Load and cache the YOLO detector model.

    Args:
        weights_path: Path to an Ultralytics-compatible YOLO weights file.

    Returns:
        Cached YOLO model instance.

    Raises:
        RuntimeError: If `ultralytics` is not installed.
    """
    global DETECTOR_MODEL, DETECTOR_MODEL_PATH

    with DETECTOR_LOCK:
        if DETECTOR_MODEL is not None and DETECTOR_MODEL_PATH == weights_path:
            return DETECTOR_MODEL

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO detection requires a working Ultralytics install in the "
                f"current Python interpreter ({sys.executable}). Original import "
                f"error: {type(exc).__name__}: {exc}"
            ) from exc

        DETECTOR_MODEL = YOLO(weights_path)
        DETECTOR_MODEL_PATH = weights_path
        return DETECTOR_MODEL


def detection_class_allowed(
    class_id: int,
    class_name: str,
    allowed_class_ids: set[int],
    allowed_class_names: set[str],
) -> bool:
    """Return whether a detector class should be treated as fish.

    Args:
        class_id: Numeric model class ID.
        class_name: Model class name.
        allowed_class_ids: Optional accepted class IDs. Empty means no ID filter.
        allowed_class_names: Optional accepted names. Empty means no name filter.

    Returns:
        `True` when the class passes all configured filters.
    """
    if allowed_class_ids and class_id not in allowed_class_ids:
        return False
    if allowed_class_names and class_name.casefold() not in allowed_class_names:
        return False
    return True


def run_fish_detection(
    raw_path: Path,
    accepted_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, Optional[str], dict]:
    """Run YOLO detection, reject bad detections, and save a padded fish crop.

    Args:
        raw_path: Raw candidate image path.
        accepted_path: Destination path for the accepted crop.
        args: Parsed CLI options containing detector settings.

    Returns:
        Tuple of `(is_valid, reject_reason, metrics)`. When valid, the cropped
        fish image has already been written to `accepted_path`.
    """
    if Image is None or ImageOps is None:
        return False, "pillow_not_installed", {"enabled": True}

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
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
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
        return True, None, metrics


def http_stream_to_file(url: str, destination: Path, retries: int = 5) -> None:
    """Stream a URL response to disk using a temporary file and retries.

    The destination is replaced atomically after a successful full download.
    Partial temporary files are removed after failed attempts.

    Args:
        url: Image URL to download.
        destination: Final file path.
        retries: Number of attempts before failing.

    Raises:
        RuntimeError: If all attempts fail.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        tmp_file = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                with open(tmp_file, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp_file.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError, IncompleteRead) as exc:
            last_error = exc
            if tmp_file.exists():
                tmp_file.unlink()
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Streaming download failed for {url}: {last_error}")


def download_file(url: str, destination: Path, overwrite: bool, retries: int = 5) -> bool:
    """Download one file unless it already exists and overwrite is disabled.

    Args:
        url: Image URL.
        destination: Target file path.
        overwrite: Redownload even when the target exists.
        retries: Number of HTTP attempts before failing.

    Returns:
        `True` if a new download occurred, `False` if the existing file was
        reused.
    """
    if destination.exists() and not overwrite:
        return False

    http_stream_to_file(url, destination, retries=retries)
    return True


def collect_photo_jobs(
    taxon_id: int,
    species_name: str,
    canonical_name: str,
    args: argparse.Namespace,
    retries: int = 5,
) -> list[dict]:
    """Build candidate photo records for one resolved species.

    This function scans observation photos, deduplicates by iNaturalist
    `photo_id`, rewrites photo URLs to the requested size, creates stable
    filenames, and stops once the candidate limit is reached.

    Args:
        taxon_id: Resolved iNaturalist taxon ID.
        species_name: Original species name from the input file.
        canonical_name: Canonical iNaturalist taxon name.
        args: Parsed CLI options.
        retries: Number of API attempts before failing.

    Returns:
        Candidate records ready for manifest writing and download scheduling.
    """
    jobs = []
    seen_photo_ids = set()
    species_slug = slugify(canonical_name)
    term_id, term_value_id = effective_annotation_filter(args)
    candidate_limit = candidate_limit_for_args(args)

    for photo in iter_observation_photos(
        taxon_id=taxon_id,
        quality_grade=args.quality_grade,
        per_page=args.per_page,
        max_pages=args.max_pages,
        license_code=args.license_code,
        place_id=args.place_id,
        exclude_captive=args.exclude_captive,
        term_id=term_id,
        term_value_id=term_value_id,
        retries=retries,
    ):
        photo_id = photo.get("photo_id")
        raw_url = photo.get("url")
        observation_id = photo.get("observation_id")
        if not photo_id or not raw_url or photo_id in seen_photo_ids:
            continue

        seen_photo_ids.add(photo_id)
        image_url = photo_url_for_size(raw_url, args.photo_size)
        filename = (
            f"{species_slug}__obs_{observation_id}__photo_{photo_id}"
            f"{infer_extension(image_url)}"
        )
        jobs.append(
            {
                "run_id": args.run_id,
                "species_name": species_name,
                "canonical_name": canonical_name,
                "taxon_id": taxon_id,
                "observation_id": observation_id,
                "photo_id": photo_id,
                "photo_url": image_url,
                "source_photo_url": raw_url,
                "filename": filename,
                "license_code": photo.get("license_code"),
                "quality_grade": photo.get("quality_grade"),
                "place_id": args.place_id,
                "observed_on": photo.get("observed_on"),
                "time_observed_at": photo.get("time_observed_at"),
                "captive": photo.get("captive"),
                "place_guess": photo.get("place_guess"),
                "user_id": photo.get("user_id"),
                "user_login": photo.get("user_login"),
                "status": "candidate",
                "reject_reason": None,
                "scores": {},
            }
        )

        if len(jobs) >= candidate_limit:
            break

    return jobs


def download_photo_job(
    candidate: dict,
    destination: Path,
    overwrite: bool,
    sleep_seconds: float,
    retries: int = 5,
) -> dict:
    """Download one candidate photo and return an updated candidate record.

    This function is designed to run inside a thread pool. It downloads or
    reuses the raw image file and annotates the candidate with raw path,
    download status, and error fields.

    Args:
        candidate: Candidate photo record from `collect_photo_jobs`.
        destination: Raw image destination path.
        overwrite: Redownload even when `destination` exists.
        sleep_seconds: Optional delay after the download attempt.
        retries: Number of HTTP attempts before failing.

    Returns:
        Candidate record with `raw_path`, `download_status`, and
        `download_error` fields.
    """
    did_download = download_file(
        url=candidate["photo_url"],
        destination=destination,
        overwrite=overwrite,
        retries=retries,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    status = "downloaded" if did_download else "skipped"
    result = dict(candidate)
    result.update(
        {
            "raw_path": str(destination),
            "download_status": status,
            "download_error": None,
        }
    )
    return result


def download_species_images(
    species_name: str,
    args: argparse.Namespace,
    output_dir: Path,
    raw_dir: Path,
    manifest_dir: Path,
) -> None:
    """Download, validate, and manifest images for one species.

    The species is resolved to an iNaturalist taxon, candidate photos are
    collected, raw images are downloaded concurrently, phase-2 validation is
    applied, accepted images are written to the output directory, and manifest
    rows are appended for candidates, accepted images, rejects, and summary
    counts.

    Args:
        species_name: Species name from the input species file.
        args: Parsed CLI options.
        output_dir: Directory for accepted images.
        raw_dir: Directory for raw downloaded candidate images.
        manifest_dir: Directory for JSONL and TSV manifests.
    """
    taxon_id, canonical_name = resolve_taxon_id(
        species_name, include_subspecies=args.include_subspecies, retries=args.retries
    )
    species_slug = slugify(canonical_name)
    accepted_species_dir = output_dir / species_slug
    raw_species_dir = raw_dir / species_slug
    accepted_species_dir.mkdir(parents=True, exist_ok=True)
    raw_species_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"\n[{species_name}] taxon_id={taxon_id} -> {canonical_name}")

    jobs = collect_photo_jobs(
        taxon_id=taxon_id,
        species_name=species_name,
        canonical_name=canonical_name,
        args=args,
        retries=args.retries,
    )
    if not jobs:
        safe_print(f"  no photos found")
        return

    candidates_path = manifest_dir / "candidates.jsonl"
    accepted_path = manifest_dir / "accepted.jsonl"
    rejected_path = manifest_dir / "rejected.jsonl"
    summary_path = manifest_dir / "species_summary.tsv"

    append_jsonl(
        candidates_path,
        [
            {
                **candidate,
                "raw_path": str(raw_species_dir / candidate["filename"]),
                "accepted_path": str(accepted_species_dir / candidate["filename"]),
            }
            for candidate in jobs
        ],
    )

    downloaded_by_photo_id = {}
    download_failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as executor:
        future_to_photo = {}
        for candidate in jobs:
            photo_id = candidate["photo_id"]
            destination = raw_species_dir / candidate["filename"]
            future = executor.submit(
                download_photo_job,
                candidate,
                destination,
                args.overwrite,
                args.sleep_seconds,
                args.retries,
            )
            future_to_photo[future] = photo_id

        for future in concurrent.futures.as_completed(future_to_photo):
            photo_id = future_to_photo[future]
            try:
                result = future.result()
                downloaded_by_photo_id[photo_id] = result
                safe_print(f"  {result['download_status']}: {result['filename']}")
            except Exception as exc:
                failed = next(
                    candidate for candidate in jobs if candidate["photo_id"] == photo_id
                )
                failed_record = {
                    **failed,
                    "status": "failed",
                    "download_status": "failed",
                    "download_error": str(exc),
                    "raw_path": str(raw_species_dir / failed["filename"]),
                    "accepted_path": None,
                    "reject_reason": "download_failed",
                }
                download_failures.append(failed_record)
                safe_print(f"  failed photo {photo_id}: {exc}")

    accepted_records = []
    rejected_records = []
    unused_valid = 0

    for candidate in jobs:
        downloaded = downloaded_by_photo_id.get(candidate["photo_id"])
        if downloaded is None:
            continue

        raw_path = Path(downloaded["raw_path"])
        accepted_image_path = accepted_species_dir / candidate["filename"]
        record = {
            **downloaded,
            "accepted_path": str(accepted_image_path),
            "validation": {},
            "detection": {},
        }

        if args.skip_image_validation:
            is_valid, reject_reason, metrics = True, None, {}
        else:
            is_valid, reject_reason, metrics = validate_image(raw_path, args)

        record["validation"] = metrics

        if not is_valid:
            record["status"] = "rejected"
            record["reject_reason"] = reject_reason
            rejected_records.append(record)
            safe_print(f"  rejected: {candidate['filename']} ({reject_reason})")
            continue

        if len(accepted_records) >= args.images_per_species:
            record["status"] = "unused"
            record["reject_reason"] = "accepted_target_reached"
            rejected_records.append(record)
            unused_valid += 1
            continue

        if args.enable_detection:
            is_detected, reject_reason, detection_metrics = run_fish_detection(
                raw_path=raw_path,
                accepted_path=accepted_image_path,
                args=args,
            )
            record["detection"] = detection_metrics
            if not is_detected:
                record["status"] = "rejected"
                record["reject_reason"] = reject_reason
                rejected_records.append(record)
                safe_print(f"  rejected: {candidate['filename']} ({reject_reason})")
                continue
            accept_status = "accepted_crop"
        else:
            accept_status = save_accepted_image(
                raw_path=raw_path,
                accepted_path=accepted_image_path,
                overwrite=args.overwrite,
            )

        record["status"] = accept_status
        record["reject_reason"] = None
        accepted_records.append(record)
        safe_print(f"  {accept_status}: {accepted_image_path.name}")

    append_jsonl(accepted_path, accepted_records)
    append_jsonl(rejected_path, [*download_failures, *rejected_records])
    append_species_summary(
        summary_path,
        {
            "run_id": args.run_id,
            "species_name": species_name,
            "canonical_name": canonical_name,
            "taxon_id": taxon_id,
            "candidates": len(jobs),
            "downloaded": len(downloaded_by_photo_id),
            "download_failed": len(download_failures),
            "accepted": len(accepted_records),
            "rejected": len(download_failures) + len(rejected_records) - unused_valid,
            "unused_valid": unused_valid,
        },
    )

    safe_print(
        f"  accepted: {len(accepted_records)}/{args.images_per_species}; "
        f"candidates: {len(jobs)}; rejected: {len(rejected_records) - unused_valid}; "
        f"unused valid: {unused_valid}; "
        f"failed: {len(download_failures)}"
    )


def main() -> None:
    """Run the full downloader CLI workflow.

    The workflow validates arguments, initializes output directories and a run
    ID, loads the species list, then processes species concurrently. Each
    species task performs candidate discovery, raw download, validation,
    accepted-image writing, and manifest updates.
    """
    args = parse_args()
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
    if not args.skip_image_validation and (Image is None or ImageOps is None):
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
            from ultralytics import YOLO  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "YOLO detection requires a working Ultralytics install in the "
                f"current Python interpreter ({sys.executable}). Original import "
                f"error: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        args.detector_class_id_set = set()
        args.detector_class_name_set = set()

    species_file = Path(args.species_file)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    manifest_dir = Path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.run_id = time.strftime("%Y%m%d-%H%M%S")

    if args.redownload:
        species_file = Path(args.redownload)
        args.overwrite = True
        safe_print(f"Redownload mode active: using {species_file} and forcing overwrite.")

    species_list = load_species(species_file)
    if not species_list:
        raise SystemExit(f"No species found in {species_file}")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.species_workers
    ) as executor:
        future_to_species = {
            executor.submit(
                download_species_images,
                species_name,
                args,
                output_dir,
                raw_dir,
                manifest_dir,
            ): species_name
            for species_name in species_list
        }
        for future in concurrent.futures.as_completed(future_to_species):
            species_name = future_to_species[future]
            try:
                future.result()
            except Exception as exc:
                safe_print(f"\n[{species_name}] failed: {exc}")


if __name__ == "__main__":
    main()
