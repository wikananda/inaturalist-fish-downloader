import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the number of accepted images in each species folder."
    )
    parser.add_argument("target", type=int, help="Required number of images per folder")
    parser.add_argument(
        "--images-dir",
        default="downloads",
        help="Directory containing one folder per species. Default: downloads",
    )
    parser.add_argument(
        "--redownload-file",
        default="redownload.txt",
        help="Where to write species below target. Default: redownload.txt",
    )
    return parser.parse_args()


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: {images_dir} directory not found.")
        return

    not_meeting_target = []

    species_folders = sorted(path for path in images_dir.iterdir() if path.is_dir())

    for folder in species_folders:
        image_count = count_images(folder)

        print(f"{folder.name}: {image_count}/{args.target}")

        if image_count < args.target:
            not_meeting_target.append(folder.name)

    print("\nFolders not meeting the target count:")
    if not_meeting_target:
        for species in not_meeting_target:
            print(species)

        redownload_path = Path(args.redownload_file)
        redownload_path.write_text(
            "\n".join(not_meeting_target) + "\n", encoding="utf-8"
        )
        print(f"\nSaved {len(not_meeting_target)} species to {redownload_path}")
    else:
        print("All folders meet the target count!")


if __name__ == "__main__":
    main()
