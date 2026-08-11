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
pip install -e '.[sam3]'
pip install -e '.[all]'
```

YOLO detection also expects trained fish detector weights, for example `models/fish-yolo.pt`.

For an SSH Linux server, use Python 3.12 when you want SAM 3 cropping:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

For a CUDA server running YOLO, CLIP, and SAM 3:

```bash
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements-server.txt
```

If your server uses a different CUDA version, install the matching PyTorch wheels from the PyTorch selector first, then run `requirements-server.txt`.

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

5. Track manual curation edits before and after inspecting `downloads/`:

```bash
inat-audit-images snapshot --images-dir downloads
```

After manually deleting or cropping images, write a change report:

```bash
inat-audit-images diff --images-dir downloads
```

The diff writes `manifests/manual_audit_changes.jsonl` with `added`, `deleted`, and `modified` records based on file hashes and image dimensions.

6. Extract Label Studio validated images into a filtered training dataset:

```bash
inat-filter-validated --csv dataset.csv --images-dir downloads --output-dir filtered_ds
```

The command copies only rows where `valid` is `validated` and writes `manifests/validated_filter_report.jsonl` with copied, skipped, missing, duplicate, and ignored rows. Use `--dry-run` to preview the report without creating files.

7. Build a final dataset from species that meet your minimum image threshold:

```bash
inat-build-final-ds 50 --images-dir filtered_ds --output-dir final_ds
```

The command copies complete-enough species folders into `final_ds/`, prints counts for included and excluded species, and writes `manifests/final_ds_included.tsv` plus `manifests/final_ds_excluded.tsv`. Use `--dry-run` to preview without copying, or `--overwrite` to replace existing folders in `final_ds/`.

8. Prepare train/validation/test folders after editing `train.txt`, `val.txt`, and `test.txt`:

```bash
inat-prepare-split --images-dir downloads --output-dir dataset_split --mode copy
```

9. Check that all species are covered by the split files:

```bash
inat-check-coverage --species-file species.txt --split-dir .
```

10. Benchmark YOLO vs SAM 3 crop quality on existing raw downloads:

```bash
inat-benchmark-croppers \
  --manifest manifests/candidates.jsonl \
  --raw-dir downloads_raw \
  --output-dir benchmarks/croppers \
  --max-images 100 \
  --max-per-species 10 \
  --backends both
```

The benchmark writes `metrics.csv`, `summary.json`, backend crops, and contact sheets without changing `downloads/` or existing manifests. SAM 3 is optional; use `--backends yolo` when SAM 3 is not installed.

To run the downloader with SAM 3 cropping instead of YOLO:

```bash
inat-download --config sam3
```

SAM saves one crop per detected fish instance. If a source photo contains multiple fish, those crops are marked `species_verification: unverified` in the accepted manifest because SAM segments fish but does not prove they are the same species as the iNaturalist observation label. The `sam3` profile uses SAM 3.1, preloads the gated Hugging Face files before any image download starts, and stores them under `models/sam3.1/`. Authenticate first with `huggingface-cli login` or set `HF_TOKEN` on SSH servers. SAM inference defaults to `sam_dtype: float32` and `sam_autocast: false` for stable CUDA runs; only enable BF16/autocast after testing your server.

## Configuration

Downloader profiles live in `configs/`. The effective config is merged in this order: `configs/default.yaml`, filter presets listed by the effective profile config, optional `--config`, then CLI overrides. Print the final merged config with:

```bash
inat-download --config smoke --print-config
```

Reusable iNaturalist observation filter presets live in `configs/filters/`. Add them to a download profile with `inat.filter_files` to experiment with quality grade, captive/alive, photo license, annotations, ordering, and raw `/observations` query parameters. The default profile excludes juvenile observations and uses commercial-safe photo licenses in this order: `cc0`, `cc-by`, then `cc-by-sa`. Sex filters are available as `not_female` and `not_male`; combine either with `not_juvenile` when you want to exclude one sex while keeping other observations.

More details are in `docs/configuration.md`, `docs/yolo_setup.md`, and `docs/clip_setup.md`.

## Broad fish baseline

Generate a taxonomy-aware species proposal without downloading images:

```bash
inat-plan-broad-species --config configs/broad_baseline_plan.yaml
```

After reviewing `plans/broad_baseline/broad_species_proposal.tsv`, run a small
pilot and then the resumable accepted-observation download:

```bash
inat-download --config broad_baseline --species-file pilot_species.txt --images-per-species 30
inat-download --config broad_baseline
```

See `docs/broad_baseline.md` for the confirmed family crosswalk, selection rules,
output files, adaptive refill behavior, and resume semantics.

See `docs/automatic_crop_quality.md` for the deterministic crop gates, SigLIP 2
calibration, optional strict SAM cascade, rejection reasons, quality-report
command, and all automated and real-model tests.
