"""CLI for tracking manual edits to accepted image folders."""

import argparse
from pathlib import Path

from ..dataset.audit import (
    build_snapshot,
    diff_snapshot,
    load_snapshot,
    save_snapshot,
    summarize_changes,
    write_changes_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse image audit CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Snapshot and diff accepted images to track manual curation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot",
        help="Record current image hashes and dimensions before manual edits.",
    )
    snapshot.add_argument(
        "--images-dir",
        default="downloads",
        help="Directory containing accepted species folders. Default: downloads",
    )
    snapshot.add_argument(
        "--snapshot",
        default="manifests/manual_audit_snapshot.json",
        help="Snapshot JSON path. Default: manifests/manual_audit_snapshot.json",
    )

    diff = subparsers.add_parser(
        "diff",
        help="Compare current images against a previous snapshot.",
    )
    diff.add_argument(
        "--images-dir",
        default="downloads",
        help="Directory containing accepted species folders. Default: downloads",
    )
    diff.add_argument(
        "--snapshot",
        default="manifests/manual_audit_snapshot.json",
        help="Snapshot JSON path. Default: manifests/manual_audit_snapshot.json",
    )
    diff.add_argument(
        "--report",
        default="manifests/manual_audit_changes.jsonl",
        help="Change report JSONL path. Default: manifests/manual_audit_changes.jsonl",
    )
    diff.add_argument(
        "--write-new-snapshot",
        default=None,
        help="Optional path to write a fresh snapshot after diffing.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the image audit command."""
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    if args.command == "snapshot":
        snapshot = build_snapshot(images_dir)
        save_snapshot(snapshot, Path(args.snapshot))
        print(f"Snapshotted {len(snapshot['files'])} images to {args.snapshot}")
        return

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot file not found: {snapshot_path}")

    snapshot = load_snapshot(snapshot_path)
    changes = diff_snapshot(snapshot, images_dir)
    write_changes_jsonl(changes, Path(args.report))
    summary = summarize_changes(changes)
    print(
        "Changes: "
        f"{summary['total']} total; "
        f"{summary['added']} added, "
        f"{summary['deleted']} deleted, "
        f"{summary['modified']} modified"
    )
    print(f"Wrote report to {args.report}")

    if args.write_new_snapshot:
        new_snapshot = build_snapshot(images_dir)
        save_snapshot(new_snapshot, Path(args.write_new_snapshot))
        print(f"Wrote fresh snapshot to {args.write_new_snapshot}")


if __name__ == "__main__":
    main()
