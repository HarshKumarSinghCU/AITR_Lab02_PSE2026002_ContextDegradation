# Human-AI Collaboration & Verification Log (`ai_assistance_log.md`)

**Student:** Harsh Kumar Singh (`PSE2026002`)

| Date | Task | AI Tool | Human Verification & Action Taken | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| Week 1 | Gap extraction & Gate G1 brief framing | Gemini | Cross-checked all extracted gaps against our updated 17-page review paper (Section 7 hypotheses H1–H5 and Table 4) and verified all 33 BibTeX references against primary sources. | Produced Gate G1 1-page research brief aligned with review terminology. |
| Week 1 | Day-One Colab sanity check script | Gemini | Inspected generated Python code (`seed=42`, `do_sample=False`, `float16` T4 memory footprint), uploaded to Google Colab, and executed on T4 GPU. | Verified runnable pipeline and exported `results/week1_sanity_output.json`. |
# Human-AI Collaboration & Verification Log (`ai_assistance_log.md`)

**Student:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad

| Date | Task | AI Tool | AI Proposal | Human Verification & Correction | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Week 1 | Gate G1 research brief & hypothesis selection | Gemini | Proposed testing H1–H5 from the review paper. | Cross-checked against updated 17-page review (Section 7 & Table 4) and verified all 33 BibTeX entries against primary sources. | Finalized Gate G1 1-page research brief. |
| Week 1 | Environment & positional sanity script | Gemini | Generated Colab script with `torch_dtype=torch.float16` and `max_new_tokens=180` on $k=5$ abstracts. | Executed on T4 GPU (`14.56 GB`); inspected console output and identified: (1) `torch_dtype` deprecation warning, (2) mid-sentence truncation at 180 tokens in Condition B, and (3) need to scale $k$ beyond 5 documents to exceed $L_{\text{eff}}$. | Logged verified output in `results/week1_sanity_output.json` and updated Week 2 protocol parameters. |
# Human-AI Collaboration & Research Integrity Audit (`ai_assistance_log.md`)

**Student:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad

## 1. Chronological Prompt & Verification Ledger
| Stage | Task | AI Proposal | Human Decision & Verification | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| Phase I | Gap & Hypothesis Framing | AI suggested 5 candidate gaps from initial 7-page draft. | Replaced with updated 17-page critical review; locked focus onto Section 7 Hypotheses H1 (qualifier loss), H3 (attribution), and H4 (contradiction flattening). | Saved `research_decisions.md` and Gate G1 brief. |
| Phase II | Day-One Sanity Script ($k=5$) | AI generated `Qwen2.5-3B-Instruct` script with `torch_dtype` and `max_new_tokens=180`. | Executed on Colab T4 GPU (`14.56 GB`). Observed: (1) `torch_dtype` deprecation warning in `transformers 5.22.1`, (2) output truncation at 180 tokens, and (3) $k=5$ fell inside $L_{\text{eff}}$. | Fixed `dtype=torch.float16`, raised `max_new_tokens=300`, and scaled $k \in \{5, 15, 25\}$. |
| Phase III | 135-Run Factorial Experiment | AI generated automated evaluation including regex citation parsing and strict `NLI >= 0.45` scoring. | Executed all 135 runs on T4 GPU. Inspected real CSV and identified why `h3_nli_faithfulness_pct` scored `0.0` in 26/27 cells: model output grouped citations (`[Doc 1, Doc 2]`) and paraphrased across papers, causing strict NLI thresholding to trigger fallback re-synthesis on every query (doubling latency to ~26s). | Documented exact diagnostic finding in `baseline_validation.md` and Section 5.3 of the manuscript without fabricating or hiding numbers. |
| Phase IV | Compute Limit Handling | Proposed scaling to `N_CLUSTERS = 15` (405 runs). | Colab T4 GPU usage limit was reached after completing the 135-run factorial dataset. Decided to freeze the verified 135-run dataset rather than fabricating unexecuted runs. | Reported 135-run results and compute constraints transparently in paper Section 6. |
