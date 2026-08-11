# Broad coral-fish family and supply experiment

Date: 2026-08-12

## Objective

Design a marine-first iNaturalist pretraining set with 150-250 fish species and,
where supply permits, 200-300 accepted independent observations per species.
The eventual product is a coral-conservation classifier applied to YOLO crops.

## Filters and observed yield

All supply counts use the same global iNaturalist restrictions as the downloader:

- research-grade observations with photos;
- non-captive observations;
- photo licences `cc0`, `cc-by`, or `cc-by-sa`;
- one accepted crop per independent observation;
- the existing detector, crop-quality, SigLIP, observer-cap, and dataset-wide
  deduplication gates.

The completed pilot accepted 34.6% of processed candidates. Species-level pass
rates varied substantially: approximately 20.0% at p10, 26.5% at p25, and 35.1%
at the median. Therefore 200 accepted observations requires about 578 source
observations at the aggregate rate, 755 at p25, and 1,000 at p10. A uniform
300-observation target is still less feasible.

## Hard iNaturalist supply ceiling

Adding families cannot create missing licensed observations. A live inventory of
all iNaturalist ray-finned fishes plus sharks and rays returned:

| Minimum licensed observations | All fish species | Expanded marine family pool |
| ---: | ---: | ---: |
| 250 | 302 | 213 |
| 350 | 186 | 132 |
| 500 | 95 | 66 |
| 600 | 60 | 37 |
| 900 | 22 | 7 |

Consequently, 150-250 species with 200-300 independent accepted observations
each is impossible from the currently permitted iNaturalist pool alone. Multiple
photos from one observation could inflate the image count but would not provide
the same independent diversity; they remain disabled.

## Family decision

The v3 configuration contains 52 coral, reef-adjacent, coastal, shark/ray, and
pelagic families. All 53 queried iNaturalist taxa were resolved as active family
taxa on the experiment date; `Epinephelidae` is queried alongside the stable
scientist-facing `Serranidae` group.

Important taxonomy decisions:

- `Anthiadidae` is active and separate, so it is included explicitly.
- `Epinephelidae` is active and separate from iNaturalist `Serranidae`; both are
  queried for the scientist-facing grouper/sea-bass scope.
- parrotfishes remain under current iNaturalist `Labridae` rather than an
  inactive `Scaridae` family query;
- fusiliers remain under current iNaturalist `Lutjanidae` rather than an
  inactive `Caesionidae` family query;
- `Aetobatidae`, `Dasyatidae`, `Mobulidae`, and `Myliobatidae` remain separate
  active ray families.

No freshwater family was added. The live marine pool reaches the lower breadth
goal without diluting coral-domain relevance. Families such as `Gobiidae`,
`Scorpaenidae`, and `Tripterygiidae` are retained with family/genus caps because
their global membership also contains non-reef or temperate species.

## Generated v3 plan

The live proposal selected:

- 3,653 candidate species inventoried;
- 213 species above the 250-observation licensed floor;
- 150 broad-training species across 44 represented families;
- 10 disjoint novel-evaluation species;
- 46 abundant scientist-requested species included in broad training;
- 68 scientist targets retained as rare-target holdouts;
- zero unmatched scientist target names.

The progressive training targets are:

| Exact target tier | Training species |
| ---: | ---: |
| 100 accepted observations | 118 |
| 200 accepted observations | 25 |
| 300 accepted observations | 7 |

The nested generated lists contain all 150 species at target 100, 32 species for
the target-200 top-up, and 7 species for the target-300 top-up. Planned total is
18,900 accepted independent observations; the pilot-calibrated projection is
18,629, so some lower-supply classes may still exhaust below their target.

## Commands

Regenerate the live count proposal when a fresh inventory is needed:

```bash
inat-plan-broad-species --config configs/broad_baseline_plan.yaml
```

Build the 100-observation base for all 150 training classes:

```bash
inat-download --config broad_baseline
```

Then resume the same per-species state and progressively top up the abundant
subsets. `images_per_species` is deliberately excluded from the resume signature,
so these target increases do not restart completed work:

```bash
inat-download \
  --config broad_baseline \
  --species-file plans/broad_coral_global_v3/target_200_train_species.txt \
  --images-per-species 200

inat-download \
  --config broad_baseline \
  --species-file plans/broad_coral_global_v3/target_300_train_species.txt \
  --images-per-species 300
```

The novel-evaluation species should be downloaded to separate output and manifest
directories so they cannot accidentally enter training.

## What is required for a uniform 200-300 target

To reach 150-250 classes at 200-300 independent observations per class, add a
second source with deployable photo licences and perform cross-source photo and
content deduplication. Alternatively, obtain explicit approval for broader
licence codes. Neither change is assumed by this plan. Loosening label, duplicate,
or multi-fish safety gates merely to reach a headline count is not recommended.
