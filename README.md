# iNaturalist Downloader

Utilities for building fish image datasets from iNaturalist. The project can extract species lists for a place and set of fish families, download observation photos, filter them with basic image checks plus optional YOLO and CLIP filtering, then prepare train/validation/test dataset folders.

## Installation

Use Python 3.10 or newer, then install the package from the repository root:

```bash
pip install -e .
```

Optional filters need extra dependencies:

```bash
pip install -e '.[yolo]'
pip install -e '.[clip]'
```

YOLO detection also expects trained fish detector weights, for example `models/fish-yolo.pt`.

## Quick Usage

1. Prepare a family list in `family.txt`, one family name per line.

2. Extract species for a place:

```bash
inat-extract-species --place Bali --families-file family.txt --output species.txt
```

This writes `species.txt` and a counts TSV beside it.

3. Download images using the default config:

```bash
inat-download --config default
```

Useful small test run:

```bash
inat-download --config smoke
```

You can override config values from the command line:

```bash
inat-download --config strict --species-file species.txt --images-per-species 30
```

Downloaded raw candidates go to `downloads_raw/`, accepted images go to `downloads/`, and audit files go to `manifests/`.

4. Check whether each species has enough accepted images:

```bash
inat-check-images 60 --images-dir downloads
```

Species below the target are written to `redownload.txt`.

5. Prepare train/validation/test folders after editing `train.txt`, `val.txt`, and `test.txt`:

```bash
inat-prepare-split --images-dir downloads --output-dir dataset_split --mode copy
```

6. Check that all species are covered by the split files:

```bash
inat-check-coverage --species-file species.txt --split-dir .
```

## Configuration

Downloader profiles live in `configs/`. The effective config is merged in this order: `configs/default.yaml`, filter presets listed by the effective profile config, optional `--config`, then CLI overrides. Print the final merged config with:

```bash
inat-download --config smoke --print-config
```

Reusable iNaturalist observation filter presets live in `configs/filters/`. Add them to a download profile with `inat.filter_files` to experiment with quality grade, captive/alive, photo license, annotations, ordering, and raw `/observations` query parameters. The default profile excludes juvenile observations and uses commercial-safe photo licenses in this order: `cc0`, `cc-by`, then `cc-by-sa`.

More details are in `docs/configuration.md`, `docs/yolo_setup.md`, and `docs/clip_setup.md`.
