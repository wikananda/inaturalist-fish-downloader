"""Compatibility module for image count checks."""

from .commands.check_images import main, parse_args
from .dataset.checks import count_images

__all__ = ["count_images", "main", "parse_args"]


if __name__ == "__main__":
    main()
