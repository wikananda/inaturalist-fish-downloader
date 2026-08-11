import json
import tempfile
import unittest
from pathlib import Path

from inaturalist_downloader.commands.quality_report import build_quality_report


class QualityReportTests(unittest.TestCase):
    def test_report_summarizes_reasons_species_and_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "candidates.jsonl").write_text(
                "{}\n{}\n{}\n", encoding="utf-8"
            )
            accepted = {
                "canonical_name": "Alpha fish",
                "clip": {"context_score": 0.2},
                "detection": {
                    "crop_quality": {"visual": {"edge_variance": 50.0}}
                },
            }
            rejected = [
                {
                    "canonical_name": "Alpha fish",
                    "reject_reason": "fish_touches_source_edge",
                    "clip": {},
                },
                {
                    "canonical_name": "Beta fish",
                    "reject_reason": "clip_filtered",
                    "clip": {"context_score": -0.4},
                },
            ]
            (root / "accepted.jsonl").write_text(
                json.dumps(accepted) + "\n", encoding="utf-8"
            )
            (root / "rejected.jsonl").write_text(
                "\n".join(json.dumps(record) for record in rejected) + "\n",
                encoding="utf-8",
            )

            report = build_quality_report(root)

        self.assertEqual(report["candidate_records"], 3)
        self.assertEqual(report["accepted_records"], 1)
        self.assertEqual(report["rejected_records"], 2)
        self.assertEqual(report["reject_reasons"]["clip_filtered"], 1)
        self.assertEqual(
            report["semantic_context_scores"]["accepted"]["median"], 0.2
        )
        alpha = next(row for row in report["species"] if row["species"] == "Alpha fish")
        self.assertEqual(alpha["acceptance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
