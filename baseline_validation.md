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
