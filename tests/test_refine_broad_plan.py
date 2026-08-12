import unittest

from inaturalist_downloader.commands.refine_broad_plan import refine_plan


class RefineBroadPlanTests(unittest.TestCase):
    def test_uses_exact_labels_and_holds_context_risk_out(self):
        proposal = [
            {
                "taxon_id": "10",
                "species": "Fish alpha",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "600",
                "dataset_role": "broad_pretraining",
            },
            {
                "taxon_id": "20",
                "species": "Fish beta",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "500",
                "dataset_role": "eligible_not_selected",
            },
        ]
        summary = [
            {
                "canonical_name": "Fish alpha",
                "accepted": "120",
                "scanned_candidates": "300",
            }
        ]
        accepted = [
            {
                "canonical_name": "Fish alpha",
                "requested_taxon_id": 10,
                "observation_taxon_id": 10,
                "observation_taxon_rank": "species",
                "observation_ancestor_ids": [],
            }
            for _ in range(110)
        ] + [
            {
                "canonical_name": "Fish alpha",
                "requested_taxon_id": 10,
                "observation_taxon_id": 11,
                "observation_taxon_rank": "species",
                "observation_ancestor_ids": [10],
            }
            for _ in range(10)
        ]

        rows = refine_plan(
            proposal,
            summary,
            accepted,
            projected_minimum=100,
            context_exclusions={"fish alpha"},
        )
        by_species = {row["species"]: row for row in rows}

        self.assertEqual(by_species["Fish alpha"]["exact_label_accepted"], 110)
        self.assertEqual(by_species["Fish alpha"]["metadata_mismatch_count"], 10)
        self.assertEqual(by_species["Fish alpha"]["v4_role"], "context_risk_holdout")
        self.assertEqual(by_species["Fish beta"]["v4_role"], "primary_train")
        self.assertEqual(by_species["Fish beta"]["planned_accepted_target"], 200)

    def test_reclassified_descendant_seeds_its_exact_species(self):
        proposal = [
            {
                "taxon_id": "10",
                "species": "Fish complex member",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "600",
                "dataset_role": "broad_pretraining",
            },
            {
                "taxon_id": "11",
                "species": "Fish split species",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "200",
                "dataset_role": "ineligible",
            },
        ]
        summary = [
            {
                "canonical_name": "Fish complex member",
                "accepted": "70",
                "scanned_candidates": "200",
            }
        ]
        accepted = [
            {
                "canonical_name": "Fish complex member",
                "observation_taxon_id": 11,
                "observation_taxon_rank": "species",
                "observation_ancestor_ids": [100],
            }
            for _ in range(70)
        ]

        rows = refine_plan(proposal, summary, accepted)
        by_species = {row["species"]: row for row in rows}

        self.assertEqual(by_species["Fish split species"]["exact_label_accepted"], 70)
        self.assertEqual(by_species["Fish split species"]["v4_role"], "reserve_backfill")
        self.assertEqual(
            by_species["Fish split species"]["planned_accepted_target"], 200
        )

    def test_existing_200_images_keep_a_200_target_even_if_projection_is_lower(self):
        proposal = [
            {
                "taxon_id": "10",
                "species": "Fish complete",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "250",
                "dataset_role": "broad_pretraining",
            }
        ]
        summary = [
            {
                "canonical_name": "Fish complete",
                "accepted": "200",
                "scanned_candidates": "1000",
            }
        ]
        accepted = [
            {
                "canonical_name": "Fish complete",
                "observation_taxon_id": 10,
                "observation_taxon_rank": "species",
                "observation_ancestor_ids": [],
            }
            for _ in range(200)
        ]

        row = refine_plan(proposal, summary, accepted)[0]

        self.assertEqual(row["planned_accepted_target"], 200)

    def test_string_observation_taxon_id_is_counted_as_exact(self):
        proposal = [
            {
                "taxon_id": "10",
                "species": "Fish exact",
                "genus": "Fish",
                "scientist_family": "Exampleidae",
                "inat_family": "Exampleidae",
                "global_count": "100",
                "dataset_role": "broad_pretraining",
            }
        ]
        summary = [
            {
                "canonical_name": "Fish exact",
                "accepted": "1",
                "scanned_candidates": "1",
            }
        ]
        accepted = [
            {
                "canonical_name": "Fish exact",
                "observation_taxon_id": "10",
                "observation_taxon_rank": "species",
                "observation_ancestor_ids": [],
            }
        ]

        row = refine_plan(proposal, summary, accepted)[0]

        self.assertEqual(row["exact_label_accepted"], 1)
        self.assertEqual(row["metadata_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
