"""Compatibility module for split coverage checks."""

from .commands.check_coverage import main, parse_args
from .dataset.checks import load_species_set as load_file

__all__ = ["load_file", "main", "parse_args"]


if __name__ == "__main__":
    main()
