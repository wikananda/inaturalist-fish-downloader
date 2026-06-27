import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inaturalist_downloader.common.inat import iter_observation_photos
from inaturalist_downloader.commands.download import download_species_images
from inaturalist_downloader.download.cli import parse_args, validate_args
from inaturalist_downloader.download.detection import DetectionOutput


class DownloadFilterConfigTests(unittest.TestCase):
    def test_profile_filter_file_merges_before_profile_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filter_path = temp_path / "alive.yaml"
            profile_path = temp_path / "profile.yaml"
            filter_path.write_text(
                "inat:\n"
                "  alive_only: true\n"
                "  order_by: created_at\n"
                "  query_params:\n"
                "    licensed: true\n",
                encoding="utf-8",
            )
            profile_path.write_text(
                "inat:\n"
                "  filter_files:\n"
                "    - alive.yaml\n"
                "  order_by: votes\n",
                encoding="utf-8",
            )

            argv = ["inat-download", "--config", str(profile_path)]
            with patch.object(sys, "argv", argv):
                args = parse_args()

        self.assertTrue(args.alive_only)
        self.assertEqual(args.order_by, "votes")
        self.assertEqual(args.query_params, {"licensed": True})

    def test_iter_observation_photos_includes_normalized_raw_query_params(self):
        payload = {
            "results": [
                {
                    "id": 123,
                    "quality_grade": "research",
                    "photos": [{"id": 456, "url": "https://example.test/square.jpg"}],
                    "user": {"id": 789, "login": "observer"},
                }
            ]
        }

        with patch("inaturalist_downloader.common.inat.api_get", return_value=payload) as api_get:
            photos = list(
                iter_observation_photos(
                    taxon_id=1,
                    quality_grade="any",
                    per_page=10,
                    max_pages=1,
                    license_code=None,
                    place_id=None,
                    exclude_captive=False,
                    term_id=None,
                    term_value_id=None,
                    order_by="created_at",
                    order="asc",
                    query_params={
                        "licensed": True,
                        "photo_license": ["cc-by", "cc-by-nc"],
                        "ignored": None,
                    },
                )
            )

        self.assertEqual(len(photos), 1)
        api_get.assert_called_once()
        params = api_get.call_args.kwargs
        self.assertEqual(params["order_by"], "created_at")
        self.assertEqual(params["order"], "asc")
        self.assertEqual(params["licensed"], "true")
        self.assertEqual(params["photo_license"], "cc-by,cc-by-nc")
        self.assertNotIn("ignored", params)

    def test_validate_args_rejects_protected_raw_query_params(self):
        args = argparse.Namespace(
            images_per_species=1,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            species_workers=1,
            download_workers=1,
            term_value_id=None,
            term_id=None,
            alive_only=False,
            query_params={"page": 2},
            blocked_license_codes=[],
            license_preference=[],
            license_code=None,
            skip_image_validation=True,
            min_width=0,
            min_height=0,
            min_file_size_kb=0,
            max_aspect_ratio=0,
            min_intensity_range=0,
            enable_detection=False,
            enable_clip_filter=False,
        )

        with self.assertRaisesRegex(SystemExit, "protected parameters: page"):
            validate_args(args)

    def test_validate_args_rejects_blocked_license_preference(self):
        args = argparse.Namespace(
            images_per_species=1,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            species_workers=1,
            download_workers=1,
            term_value_id=None,
            term_id=None,
            alive_only=False,
            query_params={},
            blocked_license_codes=["cc-by-nc"],
            license_preference=["cc0", "cc-by-nc"],
            license_code=None,
            skip_image_validation=True,
            min_width=0,
            min_height=0,
            min_file_size_kb=0,
            max_aspect_ratio=0,
            min_intensity_range=0,
            enable_detection=False,
            enable_clip_filter=False,
        )

        with self.assertRaisesRegex(SystemExit, "blocked licenses: cc-by-nc"):
            validate_args(args)

    def test_download_species_images_falls_back_through_license_preference(self):
        args = self._download_args(images_per_species=2)
        calls = []

        def collect_photo_jobs(**kwargs):
            calls.append(kwargs["license_code"])
            if kwargs["license_code"] == "cc0":
                return [], 2, True
            return [
                self._candidate(1, "cc-by"),
                self._candidate(2, "cc-by"),
            ], 2, True

        def download_photo_job(candidate, destination, overwrite, sleep_seconds, retries):
            return {**candidate, "raw_path": str(destination), "download_status": "downloaded"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                side_effect=collect_photo_jobs,
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
                side_effect=download_photo_job,
            ), patch(
                "inaturalist_downloader.commands.download.save_accepted_image",
                return_value="accepted",
            ):
                download_species_images(
                    "Test fish",
                    args,
                    temp_path / "downloads",
                    temp_path / "raw",
                    manifest_dir,
                )

        self.assertEqual(calls, ["cc0", "cc-by"])

    def test_download_species_images_skips_blocked_license_before_download(self):
        args = self._download_args(images_per_species=1)
        args.license_preference = ["cc0"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                return_value=([self._candidate(1, "cc-by-nc")], 2, True),
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
            ) as download_photo_job:
                download_species_images(
                    "Test fish",
                    args,
                    temp_path / "downloads",
                    temp_path / "raw",
                    manifest_dir,
                )

        download_photo_job.assert_not_called()

    def test_download_species_images_accepts_multiple_sam_outputs(self):
        args = self._download_args(images_per_species=2)
        args.enable_detection = True
        args.detection_backend = "sam3"

        def download_photo_job(candidate, destination, overwrite, sleep_seconds, retries):
            return {**candidate, "raw_path": str(destination), "download_status": "downloaded"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "downloads"
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()
            first_output = output_dir / "test_fish" / "fish__inst_1.jpg"
            second_output = output_dir / "test_fish" / "fish__inst_2.jpg"
            detection_outputs = [
                DetectionOutput(
                    accepted_path=first_output,
                    status="accepted_crop",
                    metrics={"backend": "sam3", "instance_index": 1},
                    clip_source_path=first_output,
                    instance_index=1,
                    instance_count=2,
                    species_verification="unverified",
                    created_output=True,
                ),
                DetectionOutput(
                    accepted_path=second_output,
                    status="accepted_crop",
                    metrics={"backend": "sam3", "instance_index": 2},
                    clip_source_path=second_output,
                    instance_index=2,
                    instance_count=2,
                    species_verification="unverified",
                    created_output=True,
                ),
            ]

            with patch(
                "inaturalist_downloader.commands.download.resolve_taxon_id",
                return_value=(1, "Test fish"),
            ), patch(
                "inaturalist_downloader.commands.download.collect_photo_jobs",
                return_value=([self._candidate(1, "cc0")], 2, True),
            ), patch(
                "inaturalist_downloader.commands.download.download_photo_job",
                side_effect=download_photo_job,
            ), patch(
                "inaturalist_downloader.commands.download.run_fish_detection_outputs",
                return_value=(detection_outputs, None, {"backend": "sam3"}),
            ):
                download_species_images(
                    "Test fish",
                    args,
                    output_dir,
                    temp_path / "raw",
                    manifest_dir,
                )

            accepted_lines = [
                line for line in (manifest_dir / "accepted.jsonl").read_text().splitlines()
                if line.strip()
            ]

        self.assertEqual(len(accepted_lines), 2)
        self.assertIn("fish__inst_1.jpg", accepted_lines[0])
        self.assertIn('"species_verification": "unverified"', accepted_lines[0])

    def _download_args(self, images_per_species=1):
        return argparse.Namespace(
            images_per_species=images_per_species,
            candidate_multiplier=1,
            max_candidates_per_species=None,
            per_page=10,
            max_pages=1,
            retries=1,
            run_id="test-run",
            include_subspecies=False,
            quality_grade="research",
            photo_size="small",
            place_id=None,
            exclude_captive=True,
            alive_only=False,
            term_id=None,
            term_value_id=None,
            order_by="votes",
            order="desc",
            query_params={},
            license_code=None,
            license_preference=["cc0", "cc-by", "cc-by-sa"],
            blocked_license_code_set={"cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd"},
            overwrite=False,
            sleep_seconds=0,
            download_workers=1,
            skip_image_validation=True,
            enable_detection=False,
            detection_backend="yolo",
            enable_clip_filter=False,
        )

    def _candidate(self, photo_id, license_code):
        return {
            "run_id": "test-run",
            "species_name": "Test fish",
            "canonical_name": "Test fish",
            "taxon_id": 1,
            "observation_id": photo_id,
            "photo_id": photo_id,
            "photo_url": f"https://example.test/{photo_id}.jpg",
            "source_photo_url": f"https://example.test/{photo_id}.jpg",
            "filename": f"test_fish__obs_{photo_id}__photo_{photo_id}.jpg",
            "requested_license_code": license_code,
            "license_priority": 1,
            "license_code": license_code,
            "quality_grade": "research",
            "place_id": None,
            "observed_on": None,
            "time_observed_at": None,
            "captive": False,
            "place_guess": None,
            "user_id": None,
            "user_login": None,
            "status": "candidate",
            "reject_reason": None,
            "scores": {},
        }


if __name__ == "__main__":
    unittest.main()
