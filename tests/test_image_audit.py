import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inaturalist_downloader.dataset.audit import (
    build_snapshot,
    diff_snapshot,
    save_snapshot,
    load_snapshot,
    summarize_changes,
    write_changes_jsonl,
)


class ImageAuditTests(unittest.TestCase):
    def test_diff_snapshot_tracks_deleted_modified_and_added_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads"
            species_dir = images_dir / "test_fish"
            species_dir.mkdir(parents=True)
            deleted = species_dir / "deleted.jpg"
            modified = species_dir / "modified.jpg"
            added = species_dir / "added.jpg"

            self._image(deleted, size=(20, 20), color="red")
            self._image(modified, size=(20, 20), color="green")
            snapshot = build_snapshot(images_dir)

            deleted.unlink()
            self._image(modified, size=(10, 10), color="green")
            self._image(added, size=(20, 20), color="blue")

            changes = diff_snapshot(snapshot, images_dir)
            by_path = {change["path"]: change for change in changes}

        self.assertEqual(by_path["test_fish/deleted.jpg"]["status"], "deleted")
        self.assertEqual(by_path["test_fish/added.jpg"]["status"], "added")
        self.assertEqual(by_path["test_fish/modified.jpg"]["status"], "modified")
        self.assertIn("content", by_path["test_fish/modified.jpg"]["change_types"])
        self.assertIn("dimensions", by_path["test_fish/modified.jpg"]["change_types"])
        self.assertEqual(summarize_changes(changes)["total"], 3)

    def test_snapshot_roundtrip_and_jsonl_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "downloads" / "test_fish"
            images_dir.mkdir(parents=True)
            self._image(images_dir / "fish.jpg")

            snapshot_path = root / "snapshot.json"
            report_path = root / "changes.jsonl"
            snapshot = build_snapshot(root / "downloads")
            save_snapshot(snapshot, snapshot_path)
            loaded = load_snapshot(snapshot_path)
            changes = diff_snapshot(loaded, root / "downloads")
            write_changes_jsonl(changes, report_path)
            report_text = report_path.read_text(encoding="utf-8")

        self.assertEqual(loaded["version"], 1)
        self.assertEqual(changes, [])
        self.assertEqual(report_text, "")

    def _image(self, path: Path, size=(16, 16), color="white"):
        Image.new("RGB", size, color=color).save(path)


if __name__ == "__main__":
    unittest.main()
