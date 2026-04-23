"""Helpers for building train/val/test image folder splits."""

import shutil
from pathlib import Path
from typing import Iterable

from .config import IMAGE_EXTENSIONS


def slugify_species_name(value: str) -> str:
    """Convert a species name to the downloader's class-folder slug."""
    return "_".join(value.strip().lower().split())


def load_split_species(path: Path) -> list[str]:
    """Load species names from a split text file."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    species = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        species.append(line)
    return species


def iter_image_files(folder: Path) -> Iterable[Path]:
    """Yield image files from a folder in stable sorted order."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_destination_ready(path: Path, overwrite: bool) -> None:
    """Ensure a destination path can be written."""
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Destination already exists: {path}. Use --overwrite to replace it."
        )
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def copy_flat_class_folder(src: Path, dst_split_dir: Path, overwrite: bool) -> int:
    """Copy class images directly into a split folder."""
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in iter_image_files(src):
        destination = dst_split_dir / image_path.name
        ensure_destination_ready(destination, overwrite=overwrite)
        shutil.copy2(image_path, destination)
        copied += 1
    return copied


def place_class_folder(
    src: Path,
    dst: Path,
    mode: str,
    flat: bool,
    overwrite: bool,
) -> int:
    """Copy, move, or symlink one species folder into a split directory."""
    if flat:
        if mode != "copy":
            raise ValueError("--flat only supports --mode copy")
        return copy_flat_class_folder(src, dst, overwrite=overwrite)

    ensure_destination_ready(dst, overwrite=overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return sum(1 for _ in iter_image_files(dst if mode != "symlink" else src))


def build_split(
    split_name: str,
    split_file: Path,
    images_dir: Path,
    output_dir: Path,
    mode: str,
    flat: bool,
    overwrite: bool,
) -> None:
    """Build one split directory from one split text file."""
    species_names = load_split_species(split_file)
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {split_name}: {len(species_names)} species listed in {split_file.name}")

    total_images = 0
    missing_species = []

    for species_name in species_names:
        species_slug = slugify_species_name(species_name)
        src = images_dir / species_slug
        if not src.exists():
            missing_species.append(species_name)
            print(f"[WARN] Missing source folder for '{species_name}' -> {src}")
            continue

        destination = split_output_dir if flat else split_output_dir / species_slug
        copied_count = place_class_folder(
            src=src,
            dst=destination,
            mode=mode,
            flat=flat,
            overwrite=overwrite,
        )
        total_images += copied_count
        print(f"[INFO] {split_name}: {species_slug} ({copied_count} images)")

    print(
        f"[DONE] {split_name}: {len(species_names) - len(missing_species)} species, "
        f"{total_images} images"
    )
    if missing_species:
        print(f"[DONE] {split_name}: {len(missing_species)} species missing from source")
