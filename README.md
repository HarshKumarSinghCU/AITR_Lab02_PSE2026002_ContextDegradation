# Empirical Evaluation of Context-Window Degradation in Multi-Document Research Synthesis

**Author:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad  
**Course:** AI Tools for Research — Post-Mid-Sem Implementation Study

This repository contains a controlled study of context-window degradation in multi-document scientific synthesis.

## Research question

How do corpus scale (`k ∈ {5, 15, 25}`) and target-evidence position (`start, middle, end`) affect H1 methodological-qualifier retention and H4 contradiction detection? To what extent can edge-salience ordering and claim-level NLI verification mitigate these failures?

## Experimental design

The factorial design contains 5 pilot clusters (135 records) or 15 extended-run clusters (405 records), with three corpus sizes, three positions, and three systems.

1. **Baseline 1 — Full Context:** all documents in their original order.
2. **Baseline 2 — Dense RAG:** top four passages selected by BGE dense retrieval.
3. **Proposed Figure 2:** dense ranking, alternating edge-salience ordering, generation, and DeBERTa NLI verification. Regeneration is triggered by missing/invalid sentence citations or a cited-claim contradiction score of at least 0.25.

The stimulus builder uses PubMedQA `pqa_labeled` records as source material. Study B includes an explicitly marked *synthetic contradictory replication*. Its contradiction sentence and illustrative p-value are experimental counterfactuals, not published findings.

## Repository map

```text
configs/exp_config.yaml
data/README.md
src/experiment_pipeline.py
scripts/env_sanity_check.py
scripts/run_factorial_experiment.py
scripts/plot_from_csv.py
notebooks/colab_experiment_runner.ipynb
paper/main.tex
paper/references.bib
results/README.md
research_decisions.md
baseline_validation.md
ai_assistance_log.md
REPRODUCIBILITY_AUDIT.md
tests/test_experiment_pipeline.py
```

## Reproduce in Google Colab

Select **Runtime → Change runtime type → T4 GPU**, then run:

```bash
git clone --branch feat/reproducible-experiment-pipeline https://github.com/HarshKumarSinghCU/AITR_Lab02_PSE2026002_ContextDegradation.git
cd AITR_Lab02_PSE2026002_ContextDegradation
pip install -r requirements.txt
python scripts/env_sanity_check.py
python -m unittest discover -s tests -v
python scripts/run_factorial_experiment.py --clusters 5
python scripts/plot_from_csv.py
```

Use `--clusters 15` for the extended 405-condition run when the Colab quota permits. The first run downloads the selected model weights and PubMedQA data.

## Outputs and metrics

The experiment writes `results/experiment_results.csv`, `results/summary_metrics.csv`, and `results/run_metadata.json`. The plotter generates figures only from the measured run-level CSV.

Metrics are headline keyword recall, binary H1 qualifier retention, binary H4 contradiction-marker detection, H3 NLI citation faithfulness, and system wall-clock latency. H1/H4 are lexical proxies, not semantic judgements. H3 evaluates citations only against documents actually supplied to the relevant system; grouped citations such as [Doc 1, Doc 2] are parsed, and citation tags are removed before NLI scoring. NLI scores are not calibrated probabilities.

## Status and research integrity

The original repository contained one-byte placeholders for the core Python files and did not include the advertised run-level results. This branch adds the implementation and reproducibility materials, but the new code has **not** been run on Colab in this repository-editing session. Historical pilot observations remain in `baseline_validation.md` and are not independently revalidated by this update.

Run the notebook, inspect the actual condition count and document manifest, and commit the measured artifacts before claiming updated findings. Do not describe Study B's synthetic counterfactual as a real publication.
