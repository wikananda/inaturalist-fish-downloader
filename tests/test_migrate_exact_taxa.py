import csv
import json
import tempfile
import unittest
from pathlib import Path

from inaturalist_downloader.commands.download import _bootstrap_progress_from_manifest
from inaturalist_downloader.commands.migrate_exact_taxa import migrate
from inaturalist_downloader.download.progress import new_progress


class ExactTaxonMigrationTests(unittest.TestCase):
    def test_migration_relabels_by_observation_taxon_and_bootstraps_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "v3" / "wrong_folder" / "fish.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"fish-image")
            source_manifest = root / "v3_accepted.jsonl"
            source_manifest.write_text(
                json.dumps(
                    {
                        "canonical_name": "Fish old label",
                        "taxon_id": 100,
                        "requested_taxon_id": 100,
                        "observation_taxon_id": 200,
                        "observation_taxon_name": "Fish correct label",
                        "observation_taxon_rank": "species",
                        "observation_id": 1,
                        "photo_id": 2,
                        "saved_output_path": str(source),
                        "filename": source.name,
                        "user_id": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = root / "plan.tsv"
            with plan.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["taxon_id", "species", "target"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {"taxon_id": 200, "species": "Fish correct label", "target": 100}
                )
            output_dir = root / "v4"
            output_manifest = root / "manifests" / "accepted.jsonl"

            report = migrate(
                source_manifest,
                root,
                plan,
                output_dir,
                output_manifest,
            )
            migrated = json.loads(output_manifest.read_text().strip())
            progress = new_progress({"taxon_id": 200}, ["global"])
            seeded = _bootstrap_progress_from_manifest(
                progress,
                output_manifest,
                taxon_id=200,
                canonical_name="Fish correct label",
            )

        self.assertEqual(report["migrated_records"], 1)
        self.assertEqual(migrated["canonical_name"], "Fish correct label")
        self.assertEqual(migrated["training_taxon_id"], 200)
        self.assertEqual(seeded, 1)
        self.assertEqual(progress.accepted_observation_ids, {1})

    def test_migration_maps_infraspecific_taxon_to_planned_parent_species(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "v3" / "fish.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"subspecies-image")
            source_manifest = root / "v3_accepted.jsonl"
            source_manifest.write_text(
                json.dumps(
                    {
                        "canonical_name": "Fish parent",
                        "observation_taxon_id": 201,
                        "observation_taxon_rank": "subspecies",
                        "observation_ancestor_ids": [10, 200],
                        "observation_id": 1,
                        "photo_id": 2,
                        "saved_output_path": str(source),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = root / "plan.tsv"
            plan.write_text(
                "taxon_id\tspecies\ttarget\n200\tFish parent\t100\n",
                encoding="utf-8",
            )

            report = migrate(
                source_manifest,
                root,
                plan,
                root / "v4",
                root / "manifest" / "accepted.jsonl",
            )

        self.assertEqual(report["newly_migrated_records"], 1)

    def test_merge_existing_preserves_primary_rows_without_duplication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = root / "v4" / "primary" / "fish.jpg"
            primary.parent.mkdir(parents=True)
            primary.write_bytes(b"primary")
            output_manifest = root / "manifest" / "accepted.jsonl"
            output_manifest.parent.mkdir(parents=True)
            output_manifest.write_text(
                json.dumps(
                    {
                        "canonical_name": "Fish primary",
                        "training_taxon_id": 100,
                        "observation_id": 1,
                        "photo_id": 2,
                        "saved_output_path": str(primary),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                json.dumps(
                    {
                        "canonical_name": "Fish reserve",
                        "observation_taxon_id": 200,
                        "observation_taxon_rank": "species",
                        "observation_ancestor_ids": [],
                        "observation_id": 3,
                        "photo_id": 4,
                        "saved_output_path": str(primary),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = root / "plan.tsv"
            plan.write_text(
                "taxon_id\tspecies\ttarget\n200\tFish reserve\t100\n",
                encoding="utf-8",
            )

            report = migrate(
                source_manifest,
                root,
                plan,
                root / "v4",
                output_manifest,
                merge_existing=True,
            )
            rows = [json.loads(line) for line in output_manifest.read_text().splitlines()]

        self.assertEqual(report["retained_existing_records"], 1)
        self.assertEqual(report["newly_migrated_records"], 1)
        self.assertEqual(len(rows), 2)

    def test_include_manifest_limits_migration_to_cleaned_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            kept_image = source_dir / "kept.jpg"
            rejected_image = source_dir / "rejected.jpg"
            kept_image.write_bytes(b"kept")
            rejected_image.write_bytes(b"rejected")
            source_manifest = root / "accepted.jsonl"
            source_manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "canonical_name": "Fish novel",
                            "observation_taxon_id": 200,
                            "observation_taxon_rank": "species",
                            "observation_id": observation_id,
                            "photo_id": photo_id,
                            "saved_output_path": str(path),
                        }
                    )
                    for observation_id, photo_id, path in (
                        (1, 11, kept_image),
                        (2, 22, rejected_image),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            include_manifest = root / "cleaned.jsonl"
            include_manifest.write_text(
                json.dumps({"observation_id": 1, "photo_id": 11}) + "\n",
                encoding="utf-8",
            )
            plan = root / "plan.tsv"
            plan.write_text(
                "taxon_id\tspecies\ttarget\n200\tFish novel\t120\n",
                encoding="utf-8",
            )
            output_manifest = root / "manifest" / "accepted.jsonl"

            report = migrate(
                source_manifest,
                root,
                plan,
                root / "output",
                output_manifest,
                include_manifest=include_manifest,
            )
            rows = [json.loads(line) for line in output_manifest.read_text().splitlines()]

        self.assertEqual(report["include_manifest_keys"], 1)
        self.assertEqual(report["newly_migrated_records"], 1)
        self.assertEqual(report["skipped"]["not_in_include_manifest"], 1)
        self.assertEqual(rows[0]["observation_id"], 1)


if __name__ == "__main__":
    unittest.main()
