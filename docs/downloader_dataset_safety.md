# Downloader dataset-safety fixes

This document records the problems found in the first broad-baseline pilot and
the protections implemented before starting the global pretraining download.

## Problems found in the pilot

1. **Species supply was overestimated.** The planner selected species from gross
   research-grade photo-observation counts (`>=500` globally), before applying
   commercial licence, detector, crop, semantic, observer, and deduplication
   gates. The pilot accepted 11,686 of 33,769 candidates (34.6%), so many species
   exhausted their usable search space at only 100-200 accepted images.
2. **The broad pretraining search was local-first.** Indonesia-first search is
   useful for local adaptation, but not for a high-capacity, reusable fish-domain
   pretraining model.
3. **The API query was trusted as licence enforcement.** One accepted pilot row
   returned `license_code: null` even though its search scope requested `cc-by`.
   A query filter is not enough; the returned photo metadata must be verified.
4. **The manifest stored the requested taxon as if it were the observed taxon.**
   It did not retain the observation's actual taxon ID, rank, or ancestry, so a
   taxon mismatch could not be audited.
5. **Deduplication existed only inside one species state.** The same photo ID
   could be accepted under two species in concurrent workers.
6. **Re-uploads under new IDs were not detected.** The pilot contained 20 exact
   duplicate pairs; five pairs crossed species labels. Some reused a photo ID,
   while others had different observation and photo IDs but identical pixels.
7. **Resume did not rebuild a dataset-wide identity index.** A new process knew
   each species' state but had no global view of accepted IDs or image content.
8. **`api_exhausted` was misleading.** It meant all configured search pages and
   scopes were consumed, not that an iNaturalist account quota had been used.
9. **Worker failures were swallowed.** A species could fail, print one line, and
   still let the command exit successfully. This produced missing completion rows
   and made interrupted runs look complete.
10. **Species summaries were append-only.** Resuming a completed species could
    leave several stale summary rows for the same taxon.
11. **The resume signature omitted acceptance-changing settings.** Changes to
    detector device/size/classes, several SAM parameters, licence enforcement,
    taxon checks, or global deduplication could otherwise mix incompatible data.

## Implemented protections

- `inat.enforce_allowed_licenses: true` checks every returned photo licence
  against the configured allowed set before the image is downloaded. Missing,
  blocked, and unexpected licences are rejected separately.
- `inat.require_taxon_membership: true` records `observation_taxon_id`, name,
  rank, and ancestor IDs, then verifies that the observation taxon equals or
  descends from the requested species taxon.
- One thread-safe `DatasetDeduplicator` is shared by all species workers. It
  protects observation IDs, photo IDs, exact SHA-256 content, and recompressed or
  visually identical content using a 64-bit perceptual difference hash. The
  default distance is `0`: pilot calibration showed that radius `4` also matched
  two unrelated cross-species pairs.
- The deduplicator loads `accepted.jsonl` on startup and computes missing hashes
  from legacy accepted paths when possible, so the same protections apply after
  restart.
- Duplicate reasons distinguish within-label duplication from cross-label
  conflict, for example `duplicate_exact_content` versus
  `conflicting_exact_content`.
- Accepted records now store `content_sha256`, `perceptual_dhash`, and complete
  deduplication metrics. `inat-quality-report` reports accepted ID/hash conflicts,
  unsafe or missing licences, safety rejections, worker failures, and stop reasons.
- Candidate and accepted rows include explicit observation/photo source URLs,
  attribution name, actual returned licence, and Creative Commons licence URL.
  Each manifest directory also stores `effective_config.yaml` plus append-only
  `run_history.jsonl` so a crop can be traced to the exact run configuration.
- Exhausted species now use `stop_reason: search_space_exhausted`.
- Failed workers are appended to `failures.jsonl`, the command exits non-zero,
  and re-running the same configuration resumes completed batches.
- A later refresh can discover observations added after a species exhausted its
  original search: run with `--refresh-exhausted-scopes`. It preserves accepted
  and seen IDs, reopens scopes at page 1, and resets only the per-refresh candidate
  budget.
- `species_summary.tsv` now contains only the latest row for each taxon.
- The progress signature now covers the new safety settings and previously
  omitted detector/SAM settings. A changed acceptance pipeline requires a new
  manifest directory instead of silently mixing data.

## Global broad-pretraining policy

`configs/broad_baseline_plan.yaml` selects globally abundant species using
research-grade, non-captive photo observations whose photo licences are one of
`cc0`, `cc-by`, or `cc-by-sa`. Regional abundance is not an eligibility condition.
The v3 coral/marine family experiment uses a 250-observation licensed floor and
progressive accepted targets. Its live proposal contains 150 training species:
118 target 100 accepted observations, 25 target 200, and 7 target 300. The nested
top-up lists contain 150, 32, and 7 species respectively. This is the strongest
honest iNaturalist-only plan found under the current filters; see
`docs/broad_coral_family_experiment.md` for the hard global supply ceiling and
family rationale.

`configs/broad_baseline.yaml` writes to new v3 paths so the pilot and v2 plan
remain intact:

- `broad_coral_global_downloads/`
- `broad_coral_global_downloads_raw/`
- `manifests/broad_coral_global_v3/`
- `plans/broad_coral_global_v3/`

Generate the new plan, inspect its summary, and start the resumable download:

```bash
inat-plan-broad-species --config configs/broad_baseline_plan.yaml
inat-download --config broad_baseline --print-config
inat-download --config broad_baseline
inat-quality-report --manifest-dir manifests/broad_coral_global_v3
```

## What automation still cannot prove

- YOLO verifies that a fish is present and that the crop geometry is usable.
- SigLIP verifies generic crop context and completeness.
- Neither proves a difficult species label. A contradictory duplicate is blocked,
  but when two iNaturalist observations disagree, the first accepted label is not
  automatically guaranteed to be the biologically correct one.
- Rejecting multi-fish sources is intentionally conservative because the
  observation label cannot be reliably assigned to one detected fish. This was
  57.2% of pilot rejections; loosening it without a species-to-instance association
  model would trade yield for label noise.
- Observer/camera/site leakage must be controlled when producing train/validation
  splits. Keep `user_id`, observation ID, and content-hash groups together.
- Real SeaLens accuracy still requires a scientist-labelled, track-grouped
  benchmark. The global pretraining set improves representation quality; it does
  not replace that benchmark.

## Existing pilot data

The old `broad_downloads/` and `manifests/broad_baseline/` are evidence and
calibration data, not the clean global-v2 training source. They contain the known
licence issue and cross-species duplicate conflicts. Do not merge them directly
into the new global dataset. Because the safety settings change the resume
signature, use the new global-v2 paths rather than forcing old state to resume.
