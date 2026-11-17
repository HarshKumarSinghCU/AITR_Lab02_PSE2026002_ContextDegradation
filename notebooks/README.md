# Experiment notebooks

## Google Colab (recommended for the T4 run)

Open [colab_experiment_runner.ipynb](colab_experiment_runner.ipynb) in Google Colab, select **Runtime → Change runtime type → T4 GPU**, and run the cells from top to bottom. The notebook checks out the reproducible experiment branch, installs the requirements, records environment details, runs unit tests, and launches the five-cluster pilot.

After the pilot completes, inspect the generated CSV/JSON outputs and plots before changing the run to 15 clusters.

## Kaggle alternative

Open [kaggle_experiment_runner.ipynb](kaggle_experiment_runner.ipynb) in Kaggle. Enable Internet and select a GPU accelerator before running. The notebook installs dependencies, runs sanity checks, executes the pilot, and packages outputs.

## Week 1 sanity check

[Week1_SanityCheck_for_GitHub.ipynb](Week1_SanityCheck_for_GitHub.ipynb) is the separate preliminary environment/sanity-check notebook.

**Important:** The experiment notebooks are runners, not evidence that the experiment has already run. Commit the executed notebook and genuine CSV/JSON/figure outputs only after a successful run and audit. Do not report synthetic Study B counterfactual content as a real biomedical finding.
