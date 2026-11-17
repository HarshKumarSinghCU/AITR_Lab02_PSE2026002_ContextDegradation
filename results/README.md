# Measured results

Run `python scripts/run_factorial_experiment.py --clusters 5` to create:

- `experiment_results.csv` — one row per experimental condition, including generated text and document manifest.
- `summary_metrics.csv` — mean and standard deviation by system, corpus size and target position.
- `run_metadata.json` — seed, cluster count, expected/actual run count and dataset disclosure.
- `environment_check.json` — runtime and package information from the environment sanity check.

Then run `python scripts/plot_from_csv.py` to generate the figure PDFs and PNGs from the measured data. The plotting script exits with a clear error if the raw result file is missing; it never generates placeholder numbers.

**No newly generated measurements are committed in this branch yet.** Historical values described in the supplied project record are documented separately in `../baseline_validation.md`.
