# L3 few-shot novel-species collection V1

This plan freezes three disjoint species roles for the L3 5-, 10-, and
20-shot experiments:

| Role | Species | Use |
|---|---:|---|
| Base/meta-train snapshot | 124 | Already seen by the broad L3/XS checkpoints |
| Novel meta-validation | 10 | Tune episodic training and prototype settings |
| Novel meta-test | 30 | Final unseen-species comparison only |

Every novel row attempts 120 accepted independent observations. Sixty is the
minimum usable result for a 20-shot support set plus a fixed query set; 100-150
is preferred. The target is a stopping ceiling, not a reason to weaken licence,
taxon, duplicate, detector, or crop-quality gates.

The 30 test taxa are absent from both `base_train_species_snapshot.txt` and
`novel_meta_validation_species.tsv`. They cover 17 active iNaturalist families
and include deliberately close groups such as `Cephalopholis`, `Stegastes`, and
`Pseudanthias`. The latter make the evaluation harder but more representative
than selecting only visually unrelated species.

The `licensed_global_observations` values come from the commercial-licence V4
proposal generated on 2026-08-12. `existing_clean_observations` comes from the
automatic V4 cleanup used to build the frozen 124-class baseline. Those 30
species have 70-97 cleaned observations each, but none was used to train the
frozen broad checkpoints.

## Download validation species

The ten validation species do not have reusable V4 crops, so collect them into
their own manifest and output folders:

```bash
inat-download --config fewshot_l3_novel_val
```

These species may be inspected repeatedly while choosing few-shot
hyperparameters. They must never be moved into the novel test role.

## Seed and top up test species

When the V4 accepted manifest, its `downloads/` tree, and the classification
cleanup manifest are available, first hard-link only crops that passed cleanup:

```bash
inat-migrate-exact-taxa \
  --source-manifest manifests/v4_download/accepted.jsonl \
  --source-root . \
  --include-manifest ../sealens_classification/data/broad_coral_global_v4_clean/cleaned_manifest.jsonl \
  --plan plans/fewshot_l3_v1/novel_meta_test_species.tsv \
  --output-dir fewshot_l3_v1_test_downloads \
  --output-manifest manifests/fewshot_l3_v1_test/accepted.jsonl
```

The include manifest is joined by `observation_id` and `photo_id`, so rejected
or quarantined V4 crops are not restored. Hard links do not duplicate image
bytes. Then let the downloader bootstrap from those accepted rows and attempt
to reach 120 per species:

```bash
inat-download --config fewshot_l3_novel_test
```

If the V4 artifacts are unavailable, skip migration and run the same download
command. It will build the isolated novel-test collection from scratch.

For a later refresh after a species reports `search_space_exhausted`, preserve
the manifest and run:

```bash
inat-download --config fewshot_l3_novel_test --refresh-exhausted-scopes
```

## Assess completion

Do not require all 30 species to reach exactly 120. Report the 60, 100, and 120
tiers before choosing the final episode pool:

```bash
inat-build-final-ds 60 \
  --images-dir fewshot_l3_v1_test_downloads \
  --included-report manifests/fewshot_l3_v1_test/ge60.tsv \
  --excluded-report manifests/fewshot_l3_v1_test/lt60.tsv \
  --dry-run

inat-build-final-ds 100 \
  --images-dir fewshot_l3_v1_test_downloads \
  --included-report manifests/fewshot_l3_v1_test/ge100.tsv \
  --excluded-report manifests/fewshot_l3_v1_test/lt100.tsv \
  --dry-run

inat-build-final-ds 120 \
  --images-dir fewshot_l3_v1_test_downloads \
  --included-report manifests/fewshot_l3_v1_test/ge120.tsv \
  --excluded-report manifests/fewshot_l3_v1_test/lt120.tsv \
  --dry-run
```

Do not use a random image split for few-shot evaluation. Support and query
manifests must be grouped by observation and observer. The final novel-test
query set should remain fixed while 5-, 10-, and 20-shot support sets are drawn
from a separate support pool.
