"""Compatibility module for species extraction."""

from .commands.extract_species import main, parse_args
from .species.api import choose_best_result, http_get_json, resolve_place, resolve_taxon
from .species.extraction import fetch_species_counts, get_species_for_place
from .species.io import write_counts_tsv, write_species_list

__all__ = [
    "choose_best_result",
    "fetch_species_counts",
    "get_species_for_place",
    "http_get_json",
    "main",
    "parse_args",
    "resolve_place",
    "resolve_taxon",
    "write_counts_tsv",
    "write_species_list",
]


if __name__ == "__main__":
    main()
