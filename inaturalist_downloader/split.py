import argparse
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_IMAGES_DIR = "downloads"
DEFAULT_OUTPUT_DIR = "dataset_split"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split iNaturalist class folders into train/val/test using text files."
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Source directory containing one folder per species. Default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for train/val/test folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split-dir",
        default=".",
        help="Directory containing train.txt, val.txt, and test.txt. Default: current directory",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move", "symlink"),
        default="copy",
        help="How to place class folders into the split output. Default: copy",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Copy images directly into each split folder instead of split/class_name/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination class folder if it already exists.",
    )
    return parser.parse_args()


def slugify_species_name(value: str) -> str:
    return "_".join(value.strip().lower().split())


def load_split_species(path: Path) -> list[str]:
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
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_destination_ready(path: Path, overwrite: bool) -> None:
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


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_dir).resolve()

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    for split_name in SPLITS:
        split_file = split_dir / f"{split_name}.txt"
        build_split(
            split_name=split_name,
            split_file=split_file,
            images_dir=images_dir,
            output_dir=output_dir,
            mode=args.mode,
            flat=args.flat,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
