import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inaturalist_downloader.commands.download import (
    _license_reject_reason,
    _taxon_reject_reason,
)
from inaturalist_downloader.common.manifest import append_species_summary
from inaturalist_downloader.download.dedup import DatasetDeduplicator
from inaturalist_downloader.download.progress import new_progress


class DatasetSafetyTests(unittest.TestCase):
    def _record(self, species, observation_id, photo_id):
        return {
            "species_name": species,
            "canonical_name": species,
            "taxon_id": observation_id,
            "observation_id": observation_id,
            "photo_id": photo_id,
            "saved_output_path": None,
        }

    def _gradient(self, path: Path, *, quality: int) -> None:
        image = Image.new("RGB", (96, 64))
        image.putdata(
            [
                (x * 2, y * 3, (x + y) % 256)
                for y in range(image.height)
                for x in range(image.width)
            ]
        )
        image.save(path, quality=quality)

    def test_missing_or_unapproved_returned_license_is_rejected(self):
        args = argparse.Namespace(
            blocked_license_code_set={"cc-by-nc"},
            enforce_allowed_licenses=True,
            allowed_license_code_set={"cc0", "cc-by", "cc-by-sa"},
        )

        self.assertEqual(
            _license_reject_reason({"license_code": None}, args),
            "missing_photo_license",
        )
        self.assertEqual(
            _license_reject_reason({"license_code": "cc-by-nc"}, args),
            "blocked_license",
        )
        self.assertEqual(
            _license_reject_reason({"license_code": "cc-by-nd"}, args),
            "disallowed_photo_license",
        )
        self.assertIsNone(
            _license_reject_reason({"license_code": "cc-by"}, args)
        )

    def test_observation_taxon_must_belong_to_requested_species(self):
        args = argparse.Namespace(require_taxon_membership=True)
        base = {"taxon_id": 10, "requested_taxon_id": 10}

        self.assertEqual(
            _taxon_reject_reason(base, args), "missing_observation_taxon"
        )
        self.assertEqual(
            _taxon_reject_reason(
                {**base, "observation_taxon_id": 20, "observation_ancestor_ids": [1]},
                args,
            ),
            "observation_taxon_mismatch",
        )
        self.assertIsNone(
            _taxon_reject_reason(
                {**base, "observation_taxon_id": 20, "observation_ancestor_ids": [10]},
                args,
            )
        )

    def test_cross_species_photo_and_exact_content_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "fish.jpg"
            self._gradient(image_path, quality=90)
            registry = DatasetDeduplicator(root / "accepted.jsonl")
            first = self._record("Species alpha", 1, 101)
            second = self._record("Species beta", 2, 102)

            self.assertTrue(registry.check_and_register(first, image_path).accepted)
            source_conflict = registry.check_source_identity(
                self._record("Species beta", 3, 101)
            )
            content_conflict = registry.check_and_register(second, image_path)

        self.assertEqual(source_conflict.reason, "conflicting_photo_id")
        self.assertEqual(content_conflict.reason, "conflicting_exact_content")
        self.assertTrue(content_conflict.metrics["cross_species"])

    def test_recompressed_near_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "quality_95.jpg"
            second_path = root / "quality_55.jpg"
            self._gradient(first_path, quality=95)
            self._gradient(second_path, quality=55)
            registry = DatasetDeduplicator(
                root / "accepted.jsonl", perceptual_distance=4
            )

            first = registry.check_and_register(
                self._record("Species alpha", 1, 101), first_path
            )
            second = registry.check_and_register(
                self._record("Species beta", 2, 102), second_path
            )

        self.assertTrue(first.accepted)
        self.assertEqual(second.reason, "conflicting_near_content")

    def test_resume_registry_loads_legacy_manifest_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "fish.jpg"
            manifest_path = root / "accepted.jsonl"
            self._gradient(image_path, quality=90)
            record = self._record("Species alpha", 1, 101)
            record["saved_output_path"] = str(image_path)
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            registry = DatasetDeduplicator(manifest_path)
            decision = registry.check_source_identity(
                self._record("Species beta", 2, 101)
            )

        self.assertEqual(decision.reason, "conflicting_photo_id")

    def test_species_summary_keeps_only_latest_row_per_taxon(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "species_summary.tsv"
            base = {
                "run_id": "first",
                "species_name": "Fish alpha",
                "canonical_name": "Fish alpha",
                "taxon_id": 1,
                "accepted": 10,
            }
            append_species_summary(path, base)
            append_species_summary(path, {**base, "run_id": "second", "accepted": 20})
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertIn("second", lines[1])
        self.assertIn("20", lines[1])

    def test_refresh_reopens_scopes_without_forgetting_seen_ids(self):
        progress = new_progress({"taxon_id": 1}, ["global-cc0", "global-cc-by"])
        progress.next_pages = {"global-cc0": 5, "global-cc-by": 8}
        progress.exhausted_scopes = {"global-cc0": True, "global-cc-by": True}
        progress.seen_photo_ids = {101, 102}
        progress.accepted_observation_ids = {1}
        progress.candidates_scanned = 500
        progress.candidate_budget_scanned = 500

        progress.refresh_exhausted_scopes()

        self.assertEqual(progress.next_pages, {"global-cc0": 1, "global-cc-by": 1})
        self.assertFalse(any(progress.exhausted_scopes.values()))
        self.assertEqual(progress.seen_photo_ids, {101, 102})
        self.assertEqual(progress.accepted_observation_ids, {1})
        self.assertEqual(progress.candidates_scanned, 500)
        self.assertEqual(progress.candidate_budget_scanned, 0)
        self.assertEqual(progress.refresh_count, 1)


if __name__ == "__main__":
    unittest.main()
