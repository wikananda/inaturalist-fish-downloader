"""Compatibility module for the downloader CLI."""

from .commands.download import download_species_images, main

__all__ = ["download_species_images", "main"]


if __name__ == "__main__":
    main()
