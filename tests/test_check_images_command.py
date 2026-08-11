import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from inaturalist_downloader.commands.check_images import main


class CheckImagesCommandTests(unittest.TestCase):
    def test_below_target_output_includes_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "filtered_ds"
            low_dir = images_dir / "test_fish"
            ok_dir = images_dir / "ok_fish"
            low_dir.mkdir(parents=True)
            ok_dir.mkdir()
            self._image(low_dir / "one.jpg")
            self._image(ok_dir / "one.jpg")
            self._image(ok_dir / "two.jpg")
            redownload_file = root / "redownload.txt"

            output = io.StringIO()
            argv = [
                "inat-check-images",
                "2",
                "--images-dir",
                str(images_dir),
                "--redownload-file",
                str(redownload_file),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                main()

            text = output.getvalue()
            redownload_text = redownload_file.read_text(encoding="utf-8")

        self.assertIn("test fish: 1/2", text)
        self.assertEqual(redownload_text, "test fish\n")

    def _image(self, path: Path):
        Image.new("RGB", (8, 8), color="white").save(path)


if __name__ == "__main__":
    unittest.main()
