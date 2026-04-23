import argparse
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from http.client import IncompleteRead


API_BASE = "https://api.inaturalist.org/v1"
DEFAULT_SPECIES_FILE = "species.txt"
DEFAULT_OUTPUT_DIR = "downloads"
DEFAULT_TIMEOUT = 30
IMAGES_PER_SPECIES = 100
DEFAULT_SPECIES_WORKERS = 5
DEFAULT_DOWNLOAD_WORKERS = 10
USER_AGENT = "inaturalist-downloader/1.0"
VALID_GRADES = {"any", "research", "needs_id", "casual"}
PRINT_LOCK = threading.Lock()


def safe_print(message: str) -> None:
    with PRINT_LOCK:
        print(message)


def parse_args() -> argparse.Namespace:
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
        help=f"Directory to store downloaded images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--images-per-species",
        type=int,
        default=IMAGES_PER_SPECIES,
        help=f"Maximum number of images to download for each species. Default: {IMAGES_PER_SPECIES}",
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
        help="Requested image size variant from iNaturalist photo URLs. Default: large",
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
        help="Maximum observation pages to scan per species. Default: 50",
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
    return parser.parse_args()


def load_species(path: Path) -> list[str]:
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
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "species"


def http_get_bytes(url: str, params: Optional[dict] = None, retries: int = 5) -> bytes:
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
    payload = http_get_bytes(f"{API_BASE}{path}", params=params, retries=retries)
    return json.loads(payload.decode("utf-8"))


def resolve_taxon_id(species_name: str, include_subspecies: bool, retries: int = 5) -> tuple[int, str]:
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
    retries: int = 5,
) -> Iterator[dict]:
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

        payload = api_get("/observations", retries=retries, **params)
        results = payload.get("results", [])
        if not results:
            return

        for observation in results:
            for photo in observation.get("photos", []):
                yield {
                    "observation_id": observation.get("id"),
                    "photo_id": photo.get("id"),
                    "url": photo.get("url"),
                    "license_code": photo.get("license_code"),
                }


def photo_url_for_size(url: str, size: str) -> str:
    pattern = r"/(square|thumb|small|medium|large|original)\.(jpg|jpeg|png)$"
    return re.sub(pattern, rf"/{size}.\2", url, flags=re.IGNORECASE)


def infer_extension(url: str) -> str:
    match = re.search(r"\.(jpg|jpeg|png)(?:\?|$)", url, flags=re.IGNORECASE)
    return f".{match.group(1).lower()}" if match else ".jpg"


def http_stream_to_file(url: str, destination: Path, retries: int = 5) -> None:
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
    if destination.exists() and not overwrite:
        return False

    http_stream_to_file(url, destination, retries=retries)
    return True


def collect_photo_jobs(
    taxon_id: int,
    canonical_name: str,
    args: argparse.Namespace,
    retries: int = 5,
) -> list[tuple[int, str, Path]]:
    jobs = []
    seen_photo_ids = set()
    species_slug = slugify(canonical_name)

    for photo in iter_observation_photos(
        taxon_id=taxon_id,
        quality_grade=args.quality_grade,
        per_page=args.per_page,
        max_pages=args.max_pages,
        license_code=args.license_code,
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
        jobs.append((photo_id, image_url, Path(filename)))

        if len(jobs) >= args.images_per_species:
            break

    return jobs


def download_photo_job(
    photo_id: int,
    image_url: str,
    destination: Path,
    overwrite: bool,
    sleep_seconds: float,
    retries: int = 5,
) -> tuple[int, str, str]:
    did_download = download_file(
        url=image_url,
        destination=destination,
        overwrite=overwrite,
        retries=retries,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    status = "downloaded" if did_download else "skipped"
    return photo_id, status, destination.name


def download_species_images(
    species_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    taxon_id, canonical_name = resolve_taxon_id(
        species_name, include_subspecies=args.include_subspecies, retries=args.retries
    )
    species_dir = output_dir / slugify(canonical_name)
    species_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"\n[{species_name}] taxon_id={taxon_id} -> {canonical_name}")

    jobs = collect_photo_jobs(
        taxon_id=taxon_id,
        canonical_name=canonical_name,
        args=args,
        retries=args.retries,
    )
    if not jobs:
        safe_print(f"  no photos found")
        return

    handled = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_workers
    ) as executor:
        future_to_photo = {}
        for photo_id, image_url, relative_path in jobs:
            destination = species_dir / relative_path
            future = executor.submit(
                download_photo_job,
                photo_id,
                image_url,
                destination,
                args.overwrite,
                args.sleep_seconds,
                args.retries,
            )
            future_to_photo[future] = photo_id

        for future in concurrent.futures.as_completed(future_to_photo):
            photo_id = future_to_photo[future]
            try:
                _, status, filename = future.result()
                handled += 1
                safe_print(f"  {status}: {filename}")
            except Exception as exc:
                safe_print(f"  failed photo {photo_id}: {exc}")

    safe_print(f"  total handled: {handled}/{len(jobs)}")


def main() -> None:
    args = parse_args()
    if args.images_per_species <= 0:
        raise SystemExit("--images-per-species must be greater than 0")
    if args.per_page <= 0:
        raise SystemExit("--per-page must be greater than 0")
    if args.max_pages <= 0:
        raise SystemExit("--max-pages must be greater than 0")
    if args.species_workers <= 0:
        raise SystemExit("--species-workers must be greater than 0")
    if args.download_workers <= 0:
        raise SystemExit("--download-workers must be greater than 0")

    species_file = Path(args.species_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            executor.submit(download_species_images, species_name, args, output_dir): species_name
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
