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
