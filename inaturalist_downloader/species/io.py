"""File I/O helpers for species extraction."""

from pathlib import Path


def load_families(path: Path) -> list[str]:
    """Load family names from a newline-delimited text file."""
    if not path.exists():
        raise FileNotFoundError(f"Families file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_species_list(path: Path, species_rows: list[dict]) -> None:
    """Write the final species names to a newline-delimited text file."""
    lines = [row["name"] for row in species_rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_counts_tsv(path: Path, species_rows: list[dict]) -> None:
    """Write species metadata and observation counts to a TSV report."""
    lines = ["taxon_id\tname\trank\tcount\tpreferred_common_name"]
    for row in species_rows:
        lines.append(
            "\t".join(
                [
                    str(row["taxon_id"]),
                    row["name"],
                    row["rank"],
                    str(row["count"]),
                    row["preferred_common_name"],
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
