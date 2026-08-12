import tempfile
import unittest
from pathlib import Path

from inaturalist_downloader.species.broad_plan import (
    build_species_proposal,
    load_plan_config,
    select_species,
    write_species_proposal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BroadPlanTests(unittest.TestCase):
    def test_confirmed_taxonomy_crosswalk(self):
        config = load_plan_config(PROJECT_ROOT / "configs" / "broad_baseline_plan.yaml")
        families = {
            item["scientist_family"]: [taxon["name"] for taxon in item["inat_taxa"]]
            for item in config["families"]
        }

        self.assertEqual(families["Labridae"], ["Labridae"])
        self.assertEqual(families["Lutjanidae"], ["Lutjanidae"])
        self.assertEqual(
            families["Serranidae"],
            ["Serranidae", "Epinephelidae"],
        )
        self.assertNotIn("Anthiadidae", families["Serranidae"])

    def test_targets_are_prioritized_and_novel_species_are_held_out(self):
        planning = {
            "min_regional_observations": 20,
            "min_global_observations": 500,
            "max_species_per_scientist_family": 5,
            "max_species_per_genus": 2,
            "novel_evaluation_fraction": 0.25,
            "random_seed": 42,
        }
        rows = []
        for index, name in enumerate(
            [
                "Targetus alpha",
                "Fishus beta",
                "Otherus gamma",
                "Moreus delta",
                "Lastus epsilon",
            ],
            start=1,
        ):
            rows.append(
                {
                    "taxon_id": index,
                    "species": name,
                    "genus": name.split()[0],
                    "scientist_family": "Exampleidae",
                    "inat_family": "Exampleidae",
                    "preferred_common_name": "",
                    "regional_count": 100 - index,
                    "global_count": 1000 - index,
                }
            )
        rows.append(
            {
                "taxon_id": 99,
                "species": "Targetus rare",
                "genus": "Targetus",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "preferred_common_name": "",
                "regional_count": 5,
                "global_count": 100,
            }
        )

        selected, unmatched = select_species(
            rows,
            planning,
            {"targetus alpha", "targetus rare"},
        )
        roles = {row["species"]: row["dataset_role"] for row in selected}

        self.assertEqual(roles["Targetus alpha"], "common_target_pretraining")
        self.assertEqual(roles["Targetus rare"], "rare_target_holdout")
        self.assertEqual(
            sum(role == "novel_evaluation" for role in roles.values()),
            1,
        )
        self.assertFalse(unmatched)

    def test_build_and_write_proposal_with_fake_counts(self):
        config = {
            "planning": {
                "name": "test",
                "region_name": "Test region",
                "region_place_id": 123,
                "quality_grade": "research",
                "photos_only": True,
                "min_regional_observations": 20,
                "min_global_observations": 500,
                "max_species_per_scientist_family": 5,
                "max_species_per_genus": 2,
                "novel_evaluation_fraction": 0,
                "random_seed": 42,
                "per_page": 500,
                "max_pages": 1,
                "retries": 1,
                "sleep_seconds": 0,
            },
            "families": [
                {
                    "scientist_family": "Exampleidae",
                    "inat_taxa": [{"name": "Exampleidae", "taxon_id": 10}],
                }
            ],
        }

        def fetcher(**kwargs):
            count = 50 if kwargs["place_id"] == 123 else 900
            return {
                11: {
                    "taxon_id": 11,
                    "species": "Example fish",
                    "preferred_common_name": "Example",
                    "count": count,
                }
            }

        rows, unmatched = build_species_proposal(config, set(), fetcher=fetcher)
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = write_species_proposal(
                Path(temp_dir), rows, unmatched, config["planning"]
            )
            proposal = (Path(temp_dir) / "broad_species_proposal.tsv").read_text()
            train_names = (Path(temp_dir) / "broad_train_species.txt").read_text()
            train_plan = (Path(temp_dir) / "broad_train_species.tsv").read_text()

        self.assertIn("Example fish", proposal)
        self.assertNotIn("\r\n", proposal)
        self.assertEqual(train_names.strip(), "Example fish")
        self.assertIn("taxon_id\tspecies\ttarget", train_plan)
        self.assertIn("11\tExample fish", train_plan)
        self.assertEqual(summary["broad_train_species"], 1)
        self.assertEqual(summary["eligible_species"], 1)
        self.assertEqual(summary["max_global_observations"], 900)

    def test_global_plan_queries_licensed_counts_without_regional_gate(self):
        config = {
            "planning": {
                "name": "global-test",
                "region_name": "unused",
                "region_place_id": 6966,
                "collect_regional_counts": False,
                "require_regional_threshold": False,
                "quality_grade": "research",
                "photos_only": True,
                "exclude_captive": True,
                "photo_license_codes": ["cc0", "cc-by", "cc-by-sa"],
                "min_regional_observations": 0,
                "min_global_observations": 2000,
                "max_species_per_scientist_family": 5,
                "max_species_per_genus": 2,
                "novel_evaluation_fraction": 0,
                "random_seed": 42,
                "per_page": 500,
                "max_pages": 1,
                "retries": 1,
                "sleep_seconds": 0,
            },
            "families": [
                {
                    "scientist_family": "Exampleidae",
                    "inat_taxa": [{"name": "Exampleidae", "taxon_id": 10}],
                }
            ],
        }
        calls = []

        def fetcher(**kwargs):
            calls.append(kwargs)
            return {
                11: {
                    "taxon_id": 11,
                    "species": "Abundant fish",
                    "preferred_common_name": "Abundant",
                    "count": 2500,
                }
            }

        rows, _ = build_species_proposal(config, set(), fetcher=fetcher)

        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["place_id"])
        self.assertEqual(
            calls[0]["photo_license_codes"], ["cc0", "cc-by", "cc-by-sa"]
        )
        self.assertTrue(calls[0]["exclude_captive"])
        self.assertEqual(rows[0]["regional_count"], 0)
        self.assertTrue(rows[0]["eligible"])

    def test_progressive_accepted_targets_are_written(self):
        planning = {
            "name": "tier-test",
            "region_name": "Global",
            "region_place_id": 1,
            "min_regional_observations": 0,
            "min_global_observations": 250,
            "max_species_per_scientist_family": 10,
            "max_species_per_genus": 4,
            "novel_evaluation_fraction": 0,
            "random_seed": 42,
            "require_regional_threshold": False,
            "expected_acceptance_rate": 0.35,
            "accepted_target_tiers": [
                {
                    "min_global_observations": 250,
                    "target_accepted_observations": 100,
                },
                {
                    "min_global_observations": 600,
                    "target_accepted_observations": 200,
                },
            ],
            "novel_evaluation_accepted_target": 100,
        }
        rows = [
            {
                "taxon_id": 1,
                "species": "Abundantus fishus",
                "genus": "Abundantus",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "preferred_common_name": "",
                "regional_count": 0,
                "global_count": 700,
            },
            {
                "taxon_id": 2,
                "species": "Broadus fishus",
                "genus": "Broadus",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "preferred_common_name": "",
                "regional_count": 0,
                "global_count": 300,
            },
        ]

        selected, _ = select_species(rows, planning, set())
        by_species = {row["species"]: row for row in selected}
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = write_species_proposal(
                Path(temp_dir), selected, set(), planning
            )
            target_100 = (
                Path(temp_dir) / "target_100_train_species.txt"
            ).read_text()
            target_200 = (
                Path(temp_dir) / "target_200_train_species.txt"
            ).read_text()

        self.assertEqual(
            by_species["Abundantus fishus"]["planned_accepted_target"], 200
        )
        self.assertEqual(
            by_species["Abundantus fishus"]["estimated_accepted_observations"],
            245,
        )
        self.assertEqual(by_species["Broadus fishus"]["planned_accepted_target"], 100)
        self.assertIn("Abundantus fishus", target_100)
        self.assertIn("Broadus fishus", target_100)
        self.assertEqual(target_200.strip(), "Abundantus fishus")
        self.assertEqual(summary["planned_train_by_accepted_target"], {"100": 1, "200": 1})
        self.assertEqual(summary["planned_train_accepted_observations"], 300)


if __name__ == "__main__":
    unittest.main()
