# AITR_Lab02_PSE2026002_ContextDegradation
Empirical evaluation of context-window degradation (H1, H3, H4) and evidence-aware synthesis (Figure 2)
# Empirical Evaluation of Context-Window Degradation in Multi-Document Research Synthesis

**Author:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad  
**Course:** AI Tools for Research (Post-Mid-Sem Implementation Study)

## 1. Research Question
In multi-document scientific synthesis, how do corpus scale ($k \in \{5, 15, 25\}$) and target evidence position ($p \in \{\text{start, middle, end}\}$) affect mid-document methodological qualifier retention ($H_1$) and contradiction detection ($H_4$), and to what extent does combining edge-salience reordering with claim-level NLI verification (`Proposed_Fig2_Verified`) mitigate these failures?

## 2. Verified Environment & Dependencies
- **Hardware:** Google Colab NVIDIA Tesla T4 GPU (`14.56 GB VRAM`)
- **Software:** Python `3.13.16`, PyTorch `2.14.1+cu130`, Transformers `5.22.1`, Sentence-Transformers
- **Models:** `Qwen/Qwen2.5-3B-Instruct` (generator), `BAAI/bge-small-en-v1.5` (retriever), `cross-encoder/nli-deberta-v3-small` (verifier)
- **Dataset:** `qiaojin/PubMedQA` (`pqa_labeled` split via HuggingFace `datasets`, MIT/Apache-2.0)

## 3. Repository Structure
- `configs/exp_config.yaml` — Frozen experimental parameters (`seed=42`, `do_sample=False`, `max_new_tokens=300`)
- `notebooks/` — Executed Google Colab notebooks (Week 1 sanity check & 135-run factorial sweep)
- `results/` — Genuine experimental outputs (`week1_sanity_output.json`, `experiment_results.csv`, `summary_metrics.csv`, PDF/PNG plots)
- `paper/main.tex` — Complete two-column LaTeX manuscript with embedded vector figures and 33 verified references
- `research_decisions.md`, `baseline_validation.md`, `ai_assistance_log.md` — Provenance and audit logs

## 4. Reproduction Instructions
```bash
pip install -r requirements.txt
# 1. Run 2-condition sanity check (takes ~60s on T4 GPU)
python scripts/week1_sanity_check.py
# 2. Run 135-run factorial experiment (takes ~8 mins on T4 GPU)
python scripts/run_factorial_experiment.py
# 3. Regenerate plots on CPU from saved summary metrics (takes 2s on CPU)
python scripts/plot_from_csv.py
