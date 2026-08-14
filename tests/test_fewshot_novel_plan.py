import csv
import json
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from inaturalist_downloader.common.utils import load_species_requests


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "plans" / "fewshot_l3_v1"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class FewshotNovelPlanTests(unittest.TestCase):
    def test_species_roles_are_frozen_and_disjoint(self):
        base = {
            line.strip().casefold()
            for line in (PLAN_DIR / "base_train_species_snapshot.txt").read_text().splitlines()
            if line.strip()
        }
        validation = _read_tsv(PLAN_DIR / "novel_meta_validation_species.tsv")
        test = _read_tsv(PLAN_DIR / "novel_meta_test_species.tsv")
        validation_names = {row["species"].casefold() for row in validation}
        test_names = {row["species"].casefold() for row in test}

        self.assertEqual(len(base), 124)
        self.assertEqual(len(validation), 10)
        self.assertEqual(len(test), 30)
        self.assertTrue(base.isdisjoint(validation_names))
        self.assertTrue(base.isdisjoint(test_names))
        self.assertTrue(validation_names.isdisjoint(test_names))

    def test_exact_ids_targets_and_supply_are_valid(self):
        for filename, expected_rows in (
            ("novel_meta_validation_species.tsv", 10),
            ("novel_meta_test_species.tsv", 30),
        ):
            path = PLAN_DIR / filename
            rows = _read_tsv(path)
            requests = load_species_requests(path)

            self.assertEqual(len(rows), expected_rows)
            self.assertEqual(len({row["species"].casefold() for row in rows}), expected_rows)
            self.assertEqual(len({int(row["taxon_id"]) for row in rows}), expected_rows)
            self.assertTrue(all(int(row["target"]) == 120 for row in rows))
            self.assertTrue(all(request.taxon_id is not None for request in requests))
            self.assertTrue(all(request.target == 120 for request in requests))

        test_rows = _read_tsv(PLAN_DIR / "novel_meta_test_species.tsv")
        self.assertTrue(
            all(int(row["existing_clean_observations"]) >= 60 for row in test_rows)
        )
        self.assertGreaterEqual(len({row["inat_family"] for row in test_rows}), 15)

    def test_plan_metadata_matches_recent_taxonomy_proposals(self):
        sources = {}
        for path in (
            ROOT / "plans" / "broad_coral_global_v3" / "broad_species_proposal.tsv",
            ROOT / "plans" / "broad_coral_global_v4" / "broad_species_proposal.tsv",
        ):
            for row in _read_tsv(path):
                sources[(row["species"], int(row["taxon_id"]))] = row

        for filename in (
            "novel_meta_validation_species.tsv",
            "novel_meta_test_species.tsv",
        ):
            for row in _read_tsv(PLAN_DIR / filename):
                source = sources[(row["species"], int(row["taxon_id"]))]
                self.assertEqual(row["inat_family"], source["inat_family"])
                self.assertEqual(
                    int(row["licensed_global_observations"]),
                    int(source["global_count"]),
                )

    def test_profiles_point_to_isolated_exact_taxon_plans(self):
        expected = {
            "fewshot_l3_novel_val.yaml": (
                "plans/fewshot_l3_v1/novel_meta_validation_species.tsv",
                "manifests/fewshot_l3_v1_val",
            ),
            "fewshot_l3_novel_test.yaml": (
                "plans/fewshot_l3_v1/novel_meta_test_species.tsv",
                "manifests/fewshot_l3_v1_test",
            ),
        }
        for filename, (species_file, manifest_dir) in expected.items():
            config = OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs" / filename), resolve=True
            )
            self.assertEqual(config["paths"]["species_file"], species_file)
            self.assertEqual(config["paths"]["manifest_dir"], manifest_dir)
            self.assertEqual(config["download"]["images_per_species"], 120)
            self.assertEqual(config["download"]["target_unit"], "observation")
            self.assertEqual(config["download"]["max_photos_per_observation"], 1)
            self.assertTrue(config["inat"]["require_exact_species_taxon"])

    def test_summary_matches_plan(self):
        summary = json.loads((PLAN_DIR / "plan_summary.json").read_text())
        test_rows = _read_tsv(PLAN_DIR / "novel_meta_test_species.tsv")
        clean_counts = [int(row["existing_clean_observations"]) for row in test_rows]

        self.assertEqual(summary["novel_meta_test_species"], len(test_rows))
        self.assertEqual(summary["existing_clean_meta_test_images"], sum(clean_counts))
        self.assertEqual(summary["meta_test_existing_clean_min"], min(clean_counts))
        self.assertEqual(summary["meta_test_existing_clean_max"], max(clean_counts))


if __name__ == "__main__":
    unittest.main()
