"""Compatibility module for dataset split preparation."""

from .commands.prepare_split import main, parse_args
from .dataset.splitter import (
    build_split,
    copy_flat_class_folder,
    ensure_destination_ready,
    iter_image_files,
    load_split_species,
    place_class_folder,
    slugify_species_name,
)

__all__ = [
    "build_split",
    "copy_flat_class_folder",
    "ensure_destination_ready",
    "iter_image_files",
    "load_split_species",
    "main",
    "parse_args",
    "place_class_folder",
    "slugify_species_name",
]


if __name__ == "__main__":
    main()
