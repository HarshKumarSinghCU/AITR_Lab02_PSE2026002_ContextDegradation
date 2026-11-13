# Baseline Validation Report (`baseline_validation.md`)

## 1. Environment & Hardware Verification (VERIFIED)
- **Hardware:** NVIDIA Tesla T4 GPU (14.56 GB VRAM) on Google Colab
- **Software:** Python 3.13.16 | PyTorch 2.14.1+cu130 | Transformers 5.22.1
- **Model & Decoding:** `Qwen/Qwen2.5-3B-Instruct` (`float16`, deterministic greedy decoding `do_sample=False`, `seed=42`)
- **Artifact Saved:** `results/week1_sanity_output.json`

## 2. Sanity Check Observations ($k=5$ Short Abstracts)
- **Condition A (Start & End, 9.28s):** Retained H1 qualifier (`5,000 tokens` limit) and H4 contradiction (`[Doc 5]`), but exhibited minor semantic distortion (misinterpreting "headline score" as "headline generation") and distractor blending (`[Doc 2]` and `[Doc 4]`).
- **Condition B (Middle, 8.45s):** Retained H1 and H4 facts accurately, confirming that ~350 tokens falls within the model's short-context effective length $L_{\text{eff}}$ (Review Sec. 2.2). Output truncated at `max_new_tokens=180`.

## 3. Required Adjustments for Full Benchmark (Gate G2)
1. Increase `max_new_tokens` from `180` to `300` to prevent mid-sentence truncation.
2. Replace deprecated `torch_dtype` argument with `dtype` for `transformers >= 5.22`.
3. Scale context length from $k=5$ (~350 tokens) to $k \in \{5, 15, 25\}$ multi-paragraph passages (~1,500 to ~8,500 tokens) to evaluate degradation beyond $L_0$.
# Baseline Validation Report (`baseline_validation.md`) — Gate G2

**Student:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad
**Pilot Dataset:** `qiaojin/PubMedQA` (`pqa_labeled`, `N_CLUSTERS = 5`, 135 total runs across $k \in \{5, 15, 25\}$ and positions `{start, middle, end}`)
**Saved Outputs:** `results/experiment_results.csv`, `results/summary_metrics.csv`, `results/fig1_positional_degradation.pdf`, `results/fig2_method_comparison.pdf`

## 1. VERIFIED Items
- **Environment & Determinism:** Executed cleanly on NVIDIA Tesla T4 GPU (14.56 GB VRAM) using `Qwen/Qwen2.5-3B-Instruct` (`float16`, `do_sample=False`, `seed=42`), `BAAI/bge-small-en-v1.5`, and `cross-encoder/nli-deberta-v3-small`.
- **Factorial Cluster Assembly:** Unit tests confirmed exact placement of positive/qualifier paper (H1) and contradicting paper (H4) at `start` (index 0), `middle` (index $\lfloor(k-2)/2\rfloor$), and `end` (index $k-1$) amidst $k-2$ topical distractors.
- **Baseline 1 (`FullContext`) Degradation at $k=25$:** Contradiction detection dropped from 100.0% ($k=5, 15$) to 60.0% at $k=25$ `middle` (80.0% at `start`/`end`), and H1 qualifier recall averaged 66.7% (minimum 40.0% at $k=25$ `end`), confirming length and positional degradation predicted in Review Section 7 (H1, H4).
- **Proposed Figure 2 Pipeline (`Edge Reordering + NLI Verification`):** Achieved 100.0% H1 qualifier recall and 100.0% H4 contradiction detection across all 9 experimental cells, with ~2x wall-clock latency (`22.12s–29.39s` vs. `11.38s–16.62s`), validating the cost-accuracy trade-off in Review Table 3.

## 2. UNCERTAIN / NEEDS CORRECTION Items (Caught Before Full Gate G3 Run)
- **`h3_nli_faithfulness_pct` Scored `0.0` in 26/27 Cells (`NEEDS CORRECTION`):**
  - *Root Cause 1:* Citation regex `\[Doc\s*(\d+)\]` missed comma-separated multi-paper citations (e.g., `[Doc 1, Doc 2]`) generated during comparative synthesis.
  - *Root Cause 2:* Passing unstripped `[Doc X]` tags into `nli-deberta-v3-small` and requiring raw `P(entailment) >= 0.45` against long concatenated abstracts penalized valid abstractive paraphrases as `neutral`, and triggered the Stage-B fallback LLM call unnecessarily on every query.
- **Required Fix for Gate G3 (`N_CLUSTERS = 15`):**
  1. Expand citation parser to capture all document IDs inside `[Doc 1, Doc 2]` or `[Doc 1][Doc 2]`, and strip citation brackets from the hypothesis sentence before NLI scoring.
  2. Score NLI attribution faithfulness using **Non-Contradiction / Entailment Support** (`P(entailment) > P(contradiction)` and `P(contradiction) < 0.20`), and trigger Stage-B re-synthesis only when cited document coverage is missing or `P(contradiction) >= 0.25`.
