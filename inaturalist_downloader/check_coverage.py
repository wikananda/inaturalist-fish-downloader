import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that species.txt entries are covered by train/val/test split files."
    )
    parser.add_argument(
        "--species-file",
        default="species.txt",
        help="Species list to check. Default: species.txt",
    )
    parser.add_argument(
        "--split-dir",
        default=".",
        help="Directory containing train.txt, val.txt, and test.txt. Default: current directory",
    )
    return parser.parse_args()


def load_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)

    species_all = load_file(Path(args.species_file))
    train = load_file(split_dir / "train.txt")
    val = load_file(split_dir / "val.txt")
    test = load_file(split_dir / "test.txt")

    covered = train | val | test
    missing = species_all - covered

    print(f"Total species in {args.species_file}: {len(species_all)}")
    print(f"Total covered in splits: {len(covered)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing species:")
        for species in sorted(missing):
            print(species)
    else:
        print("\nAll species are covered!")


if __name__ == "__main__":
    main()
