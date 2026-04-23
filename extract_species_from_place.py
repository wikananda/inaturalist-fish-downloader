import argparse
import json
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://api.inaturalist.org/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_OUTPUT = "species.txt"
DEFAULT_FISH_TAXON_ID = 47178
DEFAULT_MIN_OBSERVATIONS = 100
DEFAULT_PER_PAGE = 200
USER_AGENT = "inaturalist-species-extractor/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fish species from iNaturalist for a place and save them to a text file."
        )
    )
    parser.add_argument(
        "--place",
        default=None,
        help="Place name to resolve through iNaturalist, for example 'Bali'.",
    )
    parser.add_argument(
        "--place-id",
        type=int,
        default=None,
        help="Use a known iNaturalist place ID instead of resolving a place name.",
    )
    parser.add_argument(
        "--taxon-id",
        type=int,
        default=DEFAULT_FISH_TAXON_ID,
        help=(
            "Ancestor taxon to search under. Default is 47178 "
            "(Actinopterygii / ray-finned fishes)."
        ),
    )
    parser.add_argument(
        "--taxon-query",
        default=None,
        help="Optional taxon name to resolve and use instead of --taxon-id.",
    )
    parser.add_argument(
        "--quality-grade",
        default="research",
        choices=["any", "research", "needs_id", "casual"],
        help="Observation quality filter. Default: research",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_MIN_OBSERVATIONS,
        help=(
            "Minimum number of matching observations required for a species to be kept. "
            f"Default: {DEFAULT_MIN_OBSERVATIONS}"
        ),
    )
    parser.add_argument(
        "--photos-only",
        action="store_true",
        help="Require observations to have photos when counting species.",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=10,
        help="Minimum number of species required to avoid falling back to a broader location. Default: 10",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output text file for the species list. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--families-file",
        default="family.txt",
        help="File containing family names, one per line. Default: family.txt",
    )
    parser.add_argument(
        "--species-per-family",
        type=int,
        default=None,
        help="Maximum number of species to extract per family.",
    )
    parser.add_argument(
        "--counts-output",
        default=None,
        help=(
            "Optional TSV file to write species counts. "
            "Default: <output basename>_counts.tsv"
        ),
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help=f"Species count page size. Default: {DEFAULT_PER_PAGE}",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum species-count pages to fetch. Default: 100",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between paginated API requests. Default: 0",
    )
    parser.add_argument(
        "--include-lower-ranks",
        action="store_true",
        help="Keep taxa below species rank instead of filtering to species only.",
    )
    return parser.parse_args()


def http_get_json(path: str, params: Optional[dict] = None, retries: int = 5) -> dict:
    final_url = f"{API_BASE}{path}"
    if params:
        final_url = f"{final_url}?{urlencode(params, doseq=True)}"

    last_error = None
    for attempt in range(1, retries + 1):
        request = Request(final_url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(attempt, 5))

    raise RuntimeError(f"Request failed for {final_url}: {last_error}")


def choose_best_result(results: list[dict], query: str, name_keys: list[str]) -> dict:
    if not results:
        raise ValueError(f"No result found for '{query}'")

    normalized_query = query.casefold()
    exact_matches = []
    partial_matches = []

    for item in results:
        values = {str(item.get(key, "")).casefold() for key in name_keys}
        if normalized_query in values:
            exact_matches.append(item)
        elif any(normalized_query in value for value in values if value):
            partial_matches.append(item)

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]
    return results[0]


def resolve_place(place_query: str) -> tuple[int, str]:
    payload = http_get_json("/places/autocomplete", {"q": place_query, "per_page": 10})
    result = choose_best_result(
        payload.get("results", []),
        place_query,
        ["display_name", "name", "admin_level"],
    )
    place_name = str(result.get("display_name") or result.get("name") or place_query)
    return int(result["id"]), place_name


def resolve_taxon(taxon_query: str) -> tuple[int, str]:
    payload = http_get_json("/taxa/autocomplete", {"q": taxon_query, "per_page": 30})
    result = choose_best_result(
        payload.get("results", []),
        taxon_query,
        ["matched_term", "name", "preferred_common_name"],
    )
    taxon_name = str(result.get("name") or taxon_query)
    return int(result["id"]), taxon_name


def fetch_species_counts(args: argparse.Namespace, place_id: Optional[int]) -> list[dict]:
    species_rows = []
    seen_taxon_ids = set()

    for page in range(1, args.max_pages + 1):
        params = {
            "taxon_id": args.taxon_id,
            "page": page,
            "per_page": args.per_page,
        }
        if place_id is not None:
            params["place_id"] = place_id
        if args.quality_grade != "any":
            params["quality_grade"] = args.quality_grade
        if args.photos_only:
            params["photos"] = "true"

        payload = http_get_json("/observations/species_counts", params)
        results = payload.get("results", [])
        if not results:
            break

        for row in results:
            taxon = row.get("taxon") or {}
            taxon_id = taxon.get("id")
            if not taxon_id or taxon_id in seen_taxon_ids:
                continue

            rank = str(taxon.get("rank") or "")
            if not args.include_lower_ranks and rank != "species":
                continue

            count = int(row.get("count") or 0)
            if count < args.min_observations:
                continue

            seen_taxon_ids.add(taxon_id)
            species_rows.append(
                {
                    "taxon_id": int(taxon_id),
                    "name": str(taxon.get("name") or "").strip(),
                    "rank": rank,
                    "count": count,
                    "preferred_common_name": str(
                        taxon.get("preferred_common_name") or ""
                    ).strip(),
                }
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    species_rows.sort(key=lambda item: (-item["count"], item["name"]))
    return species_rows


def get_species_for_place(args: argparse.Namespace, place_id: Optional[int], place_name: str, families: list[str]) -> list[dict]:
    print(f"[INFO] Searching in: {place_name} ({place_id})")
    all_species_rows = []

    for family_name in families:
        print(f"[INFO] Processing family: {family_name}...")
        try:
            family_id, resolved_family_name = resolve_taxon(family_name)
            print(f"      Resolved to: {resolved_family_name} ({family_id})")

            # Temporarily override taxon_id for this family
            original_taxon_id = args.taxon_id
            args.taxon_id = family_id

            family_species = fetch_species_counts(args, place_id=place_id)

            # Apply species per family limit
            if args.species_per_family:
                family_species = family_species[:args.species_per_family]

            print(f"      Found {len(family_species)} species")
            all_species_rows.extend(family_species)

            # Restore original taxon_id
            args.taxon_id = original_taxon_id

        except ValueError as e:
            print(f"      [WARN] Could not resolve family '{family_name}': {e}")
            continue

    return all_species_rows


def write_species_list(path: Path, species_rows: list[dict]) -> None:
    lines = [row["name"] for row in species_rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_counts_tsv(path: Path, species_rows: list[dict]) -> None:
    lines = ["taxon_id\tname\trank\tcount\tpreferred_common_name"]
    for row in species_rows:
        lines.append(
            "\t".join(
                [
                    str(row["taxon_id"]),
                    row["name"],
                    row["rank"],
                    str(row["count"]),
                    row["preferred_common_name"],
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.place and not args.place_id:
        raise SystemExit("Provide either --place or --place-id")
    if args.min_observations <= 0:
        raise SystemExit("--min-observations must be greater than 0")
    if args.per_page <= 0:
        raise SystemExit("--per-page must be greater than 0")
    if args.max_pages <= 0:
        raise SystemExit("--max-pages must be greater than 0")
    if args.species_per_family is not None and args.species_per_family <= 0:
        raise SystemExit("--species-per-family must be greater than 0")

    # Read families from file
    try:
        with open(args.families_file, "r", encoding="utf-8") as f:
            families = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise SystemExit(f"Families file not found: {args.families_file}")

    # Define the fallback chain
    place_options = []

    # 1. Initial place
    if args.place_id:
        place_options.append((args.place_id, f"place_id={args.place_id}"))
    else:
        try:
            p_id, p_name = resolve_place(args.place)
            place_options.append((p_id, p_name))
        except ValueError as e:
            print(f"[WARN] Could not resolve initial place '{args.place}': {e}")

    # 2. Fallback to Indonesia
    try:
        indo_id, indo_name = resolve_place("Indonesia")
        place_options.append((indo_id, indo_name))
    except ValueError as e:
        print(f"[WARN] Could not resolve fallback place 'Indonesia': {e}")

    # 3. Fallback to Global (None)
    place_options.append((None, "Global"))

    final_species_rows = []
    for place_id, place_name in place_options:
        all_species_rows = get_species_for_place(args, place_id, place_name, families)
        
        if len(all_species_rows) >= args.min_species:
            print(f"[INFO] Criteria met: found {len(all_species_rows)} species in {place_name}.")
            final_species_rows = all_species_rows
            break
        else:
            print(f"[INFO] Criteria NOT met: found {len(all_species_rows)} species in {place_name} (min: {args.min_species}). Falling back...")
            final_species_rows = all_species_rows

    if len(final_species_rows) < args.min_species:
        print(f"[INFO] Final result {len(final_species_rows)} species did not meet min_species {args.min_species}, but it's the best we could do.")

    output_path = Path(args.output)
    counts_output = (
        Path(args.counts_output)
        if args.counts_output
        else output_path.with_name(f"{output_path.stem}_counts.tsv")
    )

    write_species_list(output_path, final_species_rows)
    write_counts_tsv(counts_output, final_species_rows)

    print(f"[DONE] Wrote {len(final_species_rows)} total taxa to {output_path}")
    print(f"[DONE] Wrote counts to {counts_output}")


if __name__ == "__main__":
    main()
