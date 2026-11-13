# Research Decisions Log (`research_decisions.md`)

**Student:** Harsh Kumar Singh (`PSE2026002`), IIIT Allahabad
**Project:** Context-Window Degradation and Its Impact on Long-Document Research Synthesis
**Repository:** `AITR_Lab02_PSE2026002_ContextDegradation`

| ID | Decision | Reason | Source / Evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| D-01 | Focus experimental study on Hypotheses H1 (mid-document qualifier loss) and H4 (contradiction smoothing) under controlled positional shifts. | Directly tests unmeasured synthesis failure modes identified in our critical review while separating position ($p$) from distractor count ($d$). | Updated Review Paper, Section 7 (pp. 12–13) & Table 4 (p. 15) | VERIFIED (Literature) |
| D-02 | Use Pathway B + C (Controlled factorial comparison + Figure 2 edge-reordering & claim-level verification). | Avoids prohibitive pre-training costs while directly evaluating the evidence-aware pipeline proposed in Figure 2 of our review. | Assignment Handout Sec. 3 (p. 2); Review Paper Sec. 8.1 & Fig. 2 (pp. 13–14) | VERIFIED (Design) |
| D-03 | Use `Qwen/Qwen2.5-3B-Instruct` (float16, greedy decoding, seed 42) on Google Colab T4 GPU (16 GB VRAM). | Fits cleanly in 16 GB VRAM without quantization dependency errors and enforces deterministic greedy decoding (`do_sample=False`). | Colab Hardware Spec & Assignment Sec. 2 Reproducibility Rule | VERIFIED (Config) |
