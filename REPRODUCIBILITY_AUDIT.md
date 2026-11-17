# Repository Reproducibility Audit

**Review date:** 18 November 2026  
**Branch:** `feat/reproducible-experiment-pipeline`  
**Reference:** supplied 64-page `geminiatr.pdf`

## Initial repository state

The repository contained the top-level project README, configuration, and provenance notes, but the main implementation paths were placeholders:

- `src/experiment_pipeline.py` and `scripts/env_sanity_check.py` contained one newline each.
- The `notebooks/`, `paper/`, and `results/` directories each had an empty README placeholder.
- There was no checked-in factorial run CSV, summary CSV, environment JSON, notebook, or manuscript source despite README claims describing them.

## Changes in this branch

- Implemented a PubMedQA loader and controlled factorial stimulus constructor.
- Added deterministic paired Study A / Study B stimuli with (k-2) topical distractors and target-pair placement at start, middle, and end.
- Marked the contradictory Study B outcome and p-value as synthetic controlled content, not a real published result.
- Implemented Full Context, top-4 Dense RAG, and the Figure 2 edge-salience + NLI verification systems.
- Added headline, H1 qualifier, H4 contradiction, H3 NLI citation-faithfulness, and system-latency measures.
- Added environment reporting, a CLI launcher, CSV-driven plotting, a Colab notebook, repository documentation, and unit tests.
- Updated the baseline/provenance notes to make clear that historical findings have no corresponding run-level artifacts in the inspected Git tree and that the new code has not yet been executed.

## Expected run counts

| Run | Clusters | Corpus sizes | Positions | Systems | Expected records |
|---|---:|---:|---:|---:|---:|
| Pilot | 5 | 3 | 3 | 3 | 135 |
| Extended | 15 | 3 | 3 | 3 | 405 |

## Verification status

| Check | Status |
|---|---|
| Repository tree and original file contents inspected | Complete |
| Assignment protocol reviewed against PDF pages 39–42 and 54–58 | Complete |
| Implementation and documentation committed to feature branch | Complete |
| Notebook created | Complete |
| Python execution on Colab T4 | Pending |
| Real PubMedQA load and model downloads | Pending |
| 135-run pilot and CSV audit | Pending |
| 405-run extended experiment | Pending |
| Final manuscript numerical tables and PDF | Pending |

No experiment was run as part of the repository edit. The PR must not be interpreted as evidence that the runtime, models, or metrics have already been empirically validated.
