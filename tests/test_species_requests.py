import tempfile
import unittest
from pathlib import Path

from inaturalist_downloader.common.utils import load_species_requests


class SpeciesRequestTests(unittest.TestCase):
    def test_loads_legacy_name_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "species.txt"
            path.write_text("# comment\nFish alpha\nFish beta\n", encoding="utf-8")
            requests = load_species_requests(path)

        self.assertEqual([request.species for request in requests], ["Fish alpha", "Fish beta"])
        self.assertTrue(all(request.taxon_id is None for request in requests))

    def test_loads_exact_tsv_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "species.tsv"
            path.write_text(
                "taxon_id\tspecies\ttarget\n59931\tAcanthurus triostegus\t150\n",
                encoding="utf-8",
            )
            requests = load_species_requests(path)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].species, "Acanthurus triostegus")
        self.assertEqual(requests[0].taxon_id, 59931)
        self.assertEqual(requests[0].target, 150)


if __name__ == "__main__":
    unittest.main()
