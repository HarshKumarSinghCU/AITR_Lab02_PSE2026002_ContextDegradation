# Research Decisions Log

**Project:** Context-Window Degradation and Its Impact on Long-Document Research Synthesis  
**Student ID:** PSE2026002  
**Institution:** IIIT Allahabad

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D-01 | Focus on H1 qualifier retention and H4 contradiction handling under corpus-scale and position changes. | Tests two failure modes identified in the review through controlled positional shifts. | Literature-motivated |
| D-02 | Compare Full Context, Dense RAG, and the Figure 2 edge-reordering + claim-verification pipeline. | Separates context exposure, retrieval selection and evidence-aware synthesis. | Design frozen |
| D-03 | Use PubMedQA `pqa_labeled`, Qwen2.5-3B-Instruct, BGE-small-en-v1.5 and DeBERTa NLI. | Matches the implementation pathway described in the assignment handout. | Configured; pin exact revisions at run time |
| D-04 | Use greedy decoding, fixed seed 42 and max_new_tokens 300. | Reduces sampling variance and follows the documented correction to the earlier 180-token truncation issue. | Configured |
| D-05 | Evaluate headline coverage, H1 qualifier retention, H4 contradiction detection, H3 NLI citation faithfulness and latency separately. | Makes information retention, conflict handling, attribution and computation independently visible. | Metrics implemented; audit after real run |
| D-06 | Label Study B's contradictory replication as synthetic controlled content. | The assignment's cluster-generation protocol inserts a counterfactual contradiction; it is not a source-record finding. | Integrity requirement |
| D-07 | Do not fabricate missing result files or claim unexecuted 405-condition runs. | The initial repository tree did not include the advertised run-level CSV/JSON artifacts. | Integrity requirement |

## Frozen factorial design

- Corpus sizes: (k in {5,15,25})
- Target pair positions: start, middle, end
- Systems: Full Context, Dense RAG (top-4), Proposed Figure 2 pipeline
- Pilot: 5 clusters = 135 runs
- Extended run: 15 clusters = 405 runs
