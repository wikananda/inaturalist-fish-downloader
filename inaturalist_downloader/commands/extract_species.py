"""CLI entrypoint for extracting species lists from iNaturalist."""

import argparse
from pathlib import Path

from ..species.api import resolve_place, resolve_taxon
from ..species.config import (
    DEFAULT_FISH_TAXON_ID,
    DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_OUTPUT,
    DEFAULT_PER_PAGE,
)
from ..species.extraction import get_species_for_place
from ..species.io import load_families, write_counts_tsv, write_species_list


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the species extraction workflow."""
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


def validate_args(args: argparse.Namespace) -> None:
    """Validate species extraction CLI arguments."""
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


def build_place_options(args: argparse.Namespace) -> list[tuple[int | None, str]]:
    """Resolve the requested place and fallback search scopes."""
    place_options = []

    if args.place_id:
        place_options.append((args.place_id, f"place_id={args.place_id}"))
    else:
        try:
            p_id, p_name = resolve_place(args.place)
            place_options.append((p_id, p_name))
        except ValueError as e:
            print(f"[WARN] Could not resolve initial place '{args.place}': {e}")

    try:
        indo_id, indo_name = resolve_place("Indonesia")
        place_options.append((indo_id, indo_name))
    except ValueError as e:
        print(f"[WARN] Could not resolve fallback place 'Indonesia': {e}")

    place_options.append((None, "Global"))
    return place_options


def main() -> None:
    """Run the full species extraction CLI workflow."""
    args = parse_args()
    validate_args(args)

    if args.taxon_query:
        args.taxon_id, taxon_name = resolve_taxon(args.taxon_query)
        print(f"[INFO] Resolved taxon query to: {taxon_name} ({args.taxon_id})")

    try:
        families = load_families(Path(args.families_file))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    final_species_rows = []
    for place_id, place_name in build_place_options(args):
        all_species_rows = get_species_for_place(args, place_id, place_name, families)

        if len(all_species_rows) >= args.min_species:
            print(f"[INFO] Criteria met: found {len(all_species_rows)} species in {place_name}.")
            final_species_rows = all_species_rows
            break

        print(
            f"[INFO] Criteria NOT met: found {len(all_species_rows)} species in "
            f"{place_name} (min: {args.min_species}). Falling back..."
        )
        final_species_rows = all_species_rows

    if len(final_species_rows) < args.min_species:
        print(
            f"[INFO] Final result {len(final_species_rows)} species did not meet "
            f"min_species {args.min_species}, but it's the best we could do."
        )

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
