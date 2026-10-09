# Validation — 9 October 2026

## Historical data checks

49 internal-consistency checks passed against the local research files:

- Six CSV exports total 4,642 records; master and screening files each contain 3,799 records with matching title order.
- API-pass counts, three parse-error resolutions and final retained counts reconcile.
- All API KEEP/MAYBE IDs are retained; only the three recorded resolutions are added.
- Input and clustered workbook record order matches; 1,947 records split into 1,479 clustered records and 468 noise records.
- All 31 theme summaries report success. Every cluster's saved representatives match the probability/relevance sorting algorithm, with 15 per cluster and 465 total.
- Original shortlist has 45 rows, all linked to records in the corpus.
- Saved embedding array is 1,947 × 768 with finite, nonzero vectors.

`results/historical_run_summary.json` records source hashes and counts without publishing bulk abstracts or full texts.

## Release checks

Five offline regression tests passed. They cover the screening-to-clustering handoff, error exclusion, resolution-ledger identity, overwrite protection, score boundaries and invalid scores, JSON extraction and invalid theme types, representative ordering, unchanged-input resume, and rejection of changed inputs/scripts or unverified checkpoints. Python source syntax and manifest hashes are checked. The release is scanned for credential-like literals and unwanted third-party references.

Validation environment: Python 3.12.14 bundled runtime, pandas 3.0.1, numpy 2.3.5 and openpyxl 3.1.5. The historical dependency environment was not captured.

## Verification limits

No live Google API calls, paid re-screening or re-embedding were performed during publication. Google SDK compatibility and model availability were not exercised in this environment. UMAP/HDBSCAN were not rerun. These checks validate saved outputs and the tested local behaviors, not screening accuracy, exhaustive literature coverage or exact reproducibility of model responses. Deduplication was counted but not independently replayed. The three recorded resolutions are preserved decisions, not newly validated relevance judgments.
