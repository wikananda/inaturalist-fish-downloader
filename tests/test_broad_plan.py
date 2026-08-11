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

        self.assertIn("Example fish", proposal)
        self.assertEqual(train_names.strip(), "Example fish")
        self.assertEqual(summary["broad_train_species"], 1)


if __name__ == "__main__":
    unittest.main()
