# AITR Lab 02 — Complete Project-Specific Prompt Pack (Prompts 01–18)

**Project theme:** Context-window degradation and its impact on long-document research synthesis  
**Implementation direction:** Position/length diagnostics + evidence provenance verification  
**Notebook inspected:** `AITR_Lab02_PSE2026002_Week1_SanityCheck.ipynb`  
**Purpose:** Turn the literature-to-experiment gap matrix into an auditable, reproducible implementation and an evidence-grounded second research paper.

---

## Before using these prompts: project facts and integrity checks

The prompts below are tailored to the uploaded assignment instructions, the uploaded Prompt 01–02 gap-matrix document, and the uploaded Colab notebook.

### What is already present in the notebook

- Generator: `Qwen/Qwen2.5-3B-Instruct`.
- Dense retriever: `BAAI/bge-small-en-v1.5`.
- NLI verifier: `cross-encoder/nli-deberta-v3-small`.
- Dataset loader: `qiaojin/PubMedQA`, configuration `pqa_labeled`.
- Three current systems: full-context synthesis, dense RAG, and a proposed Figure-2-style pipeline using retrieval ranking, context reordering, NLI checking, and targeted re-synthesis.
- Experimental factors: corpus sizes 5, 15, and 25; target evidence positions start, middle, and end.
- The saved Gate G2 cell shows 5 clusters × 3 corpus sizes × 3 positions × 3 systems = **135 completed runs**.

### Important issues that every implementation prompt must respect

1. **Do not treat the Week-1 hand-written examples as real studies.** The first cell contains fictional-looking examples attributed to “Chen et al. (2024)” and “Patel and Gomez (2025)” with invented-looking SparseRAG scores. They are suitable only as clearly labelled toy demonstrations unless their primary sources are independently located and verified. Do not cite them as real papers.
2. **Do not treat inserted qualifiers or p-values as PubMedQA facts.** The later cells insert a subgroup qualifier (“under 65 years without renal impairment”) and a negative result (`p = 0.43`) into text assembled from PubMedQA records. These are programmatically added synthetic controls, not source-grounded findings from those PubMedQA records.
3. **The positive and negative PubMedQA records are selected independently.** The code takes a `yes` record and a `no` record by list index and places them under the same research question. They are not verified contradictory studies answering the same question. This must be corrected or the experiment must be explicitly described as a controlled synthetic stress test—not real multi-paper biomedical contradiction synthesis.
4. **Gate G3 is not demonstrated as completed in the uploaded notebook.** Its cell is configured for 15 clusters × 3 sizes × 3 positions × 3 systems = **405 runs**, but its saved output ends after loading models and passing the citation-parser test. Do not report 405 completed runs until a fresh run produces the progress log, result CSV, summary and figures.
5. **The Gate G2 metrics are preliminary.** The current headline metric uses keywords pulled from a long answer; qualifier and contradiction scores use keyword presence; and the NLI faithfulness score depends on model label ordering and sentence/document matching. These are diagnostic proxies, not fully validated measures of synthesis correctness. Validate them against manually annotated examples and the model's actual `id2label` configuration.
6. **The actual seven-page literature-review manuscript is still needed for source-level audit.** The uploaded Prompt 01–02 file contains a summary and gap matrix, not the complete original review. It flags bibliographic errors. Do not claim original-paper verification until the review and primary publications have been checked.
7. **Check project identity and metadata.** The uploaded prompt document and notebook filenames use `PSE2026002`; verify that these belong to the submitting researcher and replace all names, IDs, affiliations and paths as appropriate. Never submit another person's metadata accidentally.

### Recommended defensible research scope

Use the review's Figure 2 as the modular workflow. Concentrate on:

- **Gap 3 — stage-wise diagnosis:** distinguish retrieval failure, context-position loss, context-length/distractor interference, and synthesis/verification failure.
- **Gap 2 — claim-level provenance verification:** check whether generated claims are supported by the cited evidence and test whether a targeted re-retrieval/re-synthesis step recovers missing support.

A suitable working research question is:

> **How do evidence position and corpus size affect evidence retention and source attribution in long-document synthesis, and to what extent does a retrieval-guided, claim-verification pipeline reduce these failures compared with full-context and dense-RAG baselines, at what latency cost?**

Keep two evidence tracks separate:

- **Controlled synthetic stress-test:** artificial evidence nuggets, qualifiers and contradictions with exact gold labels. This enables controlled manipulation of position and distractor count, but conclusions apply to the synthetic test conditions only.
- **Natural-data validation:** authentic PubMedQA examples, retaining their original question, context, answer and IDs. Do not add unsupported medical facts or fabricate contradictory studies. If the dataset does not provide a multi-paper contradiction label, do not claim that it does.

---

# Phase I — From literature review to research question

## PROMPT 01 — Create an accountable research workspace and summarize the project

```text
Act as a critical research mentor, ML/NLP research engineer and reproducibility auditor. I am a postgraduate student converting a completed literature review into an implementation-based research paper for the AI Tools for Research (AITR) Lab 02 assignment.

My research topic is “Context-Window Degradation and Its Impact on Long-Document Research Synthesis.” I am attaching:
1. The original literature-review PDF, if available.
2. My Prompt 01–02 evidence-based gap-matrix document.
3. My Colab notebook: AITR_Lab02_PSE2026002_Week1_SanityCheck.ipynb.
4. The AITR Lab 02 assignment instructions.

First, inspect every supplied file before proposing changes. Summarize what each file actually contains and distinguish:
- VERIFIED: directly visible in a supplied document, notebook source, saved output, or independently checked primary source;
- INFERRED: a plausible interpretation that still needs a test;
- UNKNOWN: evidence not supplied or not checked.

Create a project workspace plan with these files:
- research_decisions.md
- evidence_ledger.csv
- gap_matrix.md
- experiment_protocol.md
- baseline_validation.md
- ai_assistance_log.md
- data/README.md
- results/README.md

For every decision, record date, decision, reason, evidence, confidence/status, and next action. Do not invent a date, source, result, code execution, citation, or repository link.

Inspect the notebook's code and saved outputs. Explicitly identify its current models, dataset, methods, experimental factors, run counts, outputs, limitations and any mismatch between configured experiments and completed saved outputs.

Pay special attention to whether any text attributed to a paper, any subgroup qualifier, any p-value, or any negative result was added synthetically by code rather than taken from a source. Label such content SYNTHETIC. Never describe it as a real published biomedical result.

Do not rewrite the notebook yet. End with:
A. A concise project-scope summary;
B. A file-by-file audit table;
C. A list of missing materials;
D. The five most urgent validity risks;
E. A prioritized action plan for Gates G1–G4.
```

**Save:** Workspace plan, research decision log, evidence ledger and initial notebook audit.

## PROMPT 02 — Extract and rank evidence-based research gaps from the review

```text
Act as a skeptical literature-review analyst. Analyze the original literature-review PDF and my attached Prompt 01–02 gap-matrix document. Do not rely on the gap matrix alone when the original review is available.

The review topic is context-window degradation in long-document and multi-document research synthesis. Extract:
1. The central research problem and operational definition of context-window degradation.
2. The taxonomy of position degradation, length/distractor degradation and task/reasoning degradation.
3. The methods discussed: long-context models, sparse attention, RAG, context compression, hierarchical synthesis and evidence-aware/claim-level verification.
4. The benchmarks and datasets discussed, including Lost in the Middle, LongBench, RULER and any others explicitly present.
5. The findings and limitations actually reported by the reviewed papers.
6. The review's open questions and conclusions.

Produce five computationally testable gaps. For each gap provide:
- Exact section, page, figure, table or cited reference in the original review;
- The limitation explicitly reported by the cited authors, if confirmed;
- My own inference, clearly labelled INFERRED;
- A narrow experimental question;
- Required data and ground-truth labels;
- Baseline/comparator;
- Primary and secondary metrics;
- A feasible experiment on a Google Colab Tesla T4-class GPU;
- Main confounders and validity risks;
- What evidence would falsify the proposed hypothesis.

Rank the five gaps by (i) significance, (ii) implementation feasibility, (iii) compute cost, (iv) availability of trustworthy data and (v) how directly they follow from the review.

Prioritize the combination of stage-wise loss diagnosis (retrieval, position, length, synthesis) and claim-level provenance verification only if the original review supports it. Do not claim novelty. Flag every bibliographic entry or factual claim that needs primary-source verification. The supplied gap-matrix document itself flags possible metadata errors in LongBench, LED, RULER, the summarization survey and Lost in the Middle; independently verify these instead of copying them.

Do not invent titles, authors, DOIs, arXiv identifiers, numerical results or limitations. If the original review PDF is absent, state that the analysis is provisional and list exactly what cannot be verified.
```

**Save:** Updated `gap_matrix.md` with source-to-gap traceability and a ranked shortlist.

## PROMPT 03 — Map the selected gaps to a runnable experiment and existing notebook

```text
Act as an implementation-focused research supervisor. Use the attached literature review, gap matrix, assignment and existing Colab notebook.

Selected direction:
- Gap 3: disentangle retrieval miss, context-position loss, length/distractor interference and final synthesis/verification failure.
- Gap 2: test claim-level provenance checking and targeted re-retrieval/re-synthesis.

Candidate research question:
“How do evidence position and corpus size affect evidence retention and source attribution in long-document synthesis, and can retrieval-guided claim verification reduce these failures compared with full-context and dense-RAG baselines, with acceptable latency overhead?”

Map the literature to an executable study. Compare:
A. Reproduction plus extension;
B. Controlled comparison;
C. Targeted modification.

For each pathway identify the minimum baseline, proposed change, dataset, test split, metrics, code modules, compute requirements, likely failure modes and evidence required. Recommend the smallest scientifically defensible study.

Audit my existing notebook line by line and map each current function to the research pipeline:
- data loading and cluster construction;
- document ordering/position manipulation;
- dense retrieval;
- full-context baseline;
- dense-RAG baseline;
- proposed reordering and NLI verification;
- targeted re-synthesis;
- metric computation;
- CSV/figure export.

Do not assume these modules are scientifically correct because they execute. Specifically assess:
1. The hand-written “Chen et al. (2024)” and “Patel and Gomez (2025)” examples in the Week-1 cell;
2. The programmatically inserted subgroup qualifier and p = 0.43 in the later cells;
3. The independent selection of positive and negative PubMedQA records;
4. The validity of treating independently selected records as contradictory papers on one question;
5. Whether the current keyword metrics measure actual evidence retention;
6. Whether NLI scores are interpreted with the correct class-label order;
7. Whether the claim-verification step is truly claim-level or is comparing a whole draft against top-ranked documents;
8. Whether retrieval recall and source attribution can be measured separately.

Give a literature-to-code mapping table and a minimum-change plan. Preserve useful code and the existing three-system comparison where appropriate. Do not rewrite everything without a reason.

Recommend whether the controlled synthetic probes should be retained as a separately labelled stress-test, while a natural PubMedQA evaluation is reported separately. State exactly what claims each track can and cannot support.
```

**Save:** Literature-to-experiment map, module map and recommended implementation pathway.

## PROMPT 04 — Verify novelty, primary sources, data access and feasibility

```text
Act as a research-integrity and feasibility auditor. Evaluate the proposed study before more coding.

Study: position- and corpus-size sensitivity in multi-document synthesis, with retrieval-guided claim-level evidence verification and targeted re-retrieval.

Search for neighboring work using scholarly sources and primary papers. Inspect work on:
- Lost in the Middle and positional sensitivity;
- long-context evaluation and context-length stress tests;
- RAG and retrieval evaluation;
- evidence-grounded or attribution-aware generation;
- NLI-based factuality/claim verification;
- iterative retrieval or corrective RAG;
- long-document and multi-document synthesis benchmarks.

For each source record the exact title, authors, year, venue/arXiv identifier, stable URL, method, dataset, metrics, limitation and relevance. Verify metadata from the publisher, proceedings or arXiv record. Do not invent citations. Separate confirmed facts from claims that still require inspection of the full paper.

Audit the planned resources:
- `qiaojin/PubMedQA`, configuration `pqa_labeled`;
- `Qwen/Qwen2.5-3B-Instruct`;
- `BAAI/bge-small-en-v1.5`;
- `cross-encoder/nli-deberta-v3-small`;
- Google Colab/Tesla T4 or CPU fallback.

Check dataset documentation, model cards, licenses/terms, model-label mappings, approximate memory needs and the exact inference requirements. Do not assume a model can fit simply because a previous cell once loaded it.

Assess whether PubMedQA is appropriate for a controlled multi-document position/contradiction experiment. If the current code creates artificial qualifiers or contradictions, propose a scientifically honest design:
A. Controlled synthetic stress-test with explicit synthetic labels and no false real-paper attribution;
B. Separate natural-data evaluation using authentic PubMedQA fields without fabricated findings;
C. Another benchmark only if you can verify access and fit.

Deliver a feasibility matrix, novelty-risk statement (no novelty guarantee), primary-source verification checklist, compute plan and go/no-go decision. Include a fallback if the generator or NLI model cannot run on the T4.
```

**Save:** Feasibility/novelty matrix, reference audit and go/no-go decision.

## PROMPT 05 — Finalize research questions, hypotheses and measurable criteria

```text
Act as a strict experimental-design mentor. Use only the verified review evidence, approved gap matrix, verified data documentation and existing notebook audit.

Develop one primary research question and no more than three supporting questions. The primary focus should combine:
- evidence position (start, middle, end);
- corpus size/distractor load (5, 15, 25 documents, if justified);
- retrieval and context construction;
- claim-level source verification and targeted re-retrieval;
- quality versus latency trade-offs.

For every research question specify:
1. Independent variables;
2. Dependent variables;
3. Controlled variables;
4. Experimental unit and sample size;
5. Data split and gold labels;
6. Baseline and proposed method;
7. Primary metric, exact formula and direction (higher/lower is better);
8. Secondary metrics;
9. A meaningful decision criterion;
10. An observation that would contradict the hypothesis.

Use candidate hypotheses only where scientifically justified, for example:
- H1: evidence position changes the probability that a gold evidence nugget is retained;
- H2: increasing distractor count changes evidence retention and/or attribution;
- H3: retrieval-guided verification changes support/attribution quality compared with static baselines;
- H4: any quality gain has a measurable latency/token overhead.

Do not assume these hypotheses will be supported. Distinguish the previously published ideas from the part this experiment actually tests. If the evaluation corpus is synthetic, explicitly limit conclusions to the controlled benchmark.

Finish with a one-page Gate G1 research brief and a metric dictionary. Do not fill unknown values with guesses.
```

**Save:** Gate G1 research brief, finalized research questions, hypotheses and metric dictionary.

---

# Phase II — Experiment design and protocol freeze

## PROMPT 06 — Validate the dataset, data construction and comparators

```text
Act as a dataset and benchmark specialist. Audit the data design before running another experiment.

Current dataset candidate: `qiaojin/PubMedQA`, `pqa_labeled`. Inspect the official dataset documentation and the actual fields used by my notebook: `question`, `context.contexts`, `final_decision`, `long_answer` and `pubid` (confirm field names from loaded records; do not assume).

Produce a dataset card for this project containing source, license/terms, size, field definitions, label distribution, missingness, inclusion/exclusion rules, preprocessing and limitations.

Audit the current data-construction procedure. The code independently selects a `yes` record and a `no` record and puts them under the same question. It also adds a subgroup qualifier and a p-value statement. Determine why this does not establish a genuine cross-paper contradiction. Propose a correction that never misattributes fabricated claims to PubMedQA sources.

Design two clearly separated evaluation tracks:
TRACK A — Controlled synthetic diagnostic:
- use explicitly labelled synthetic evidence nuggets/qualifiers/contradictions;
- define the exact gold location and source ID of every nugget;
- use neutral fictional study identifiers rather than fabricated author citations or fake PubMed IDs;
- preserve controlled variables and document generation rules;
- ensure the synthetic track is described as synthetic in every table and figure.

TRACK B — Natural-data validation:
- preserve authentic PubMedQA questions, contexts, labels, long answers and IDs;
- do not add unsupported clinical statements;
- use only tasks and labels actually supported by the dataset;
- do not claim it measures multi-paper contradiction unless valid paired evidence labels exist.

Define train/development/test or construction/validation/test partitions as appropriate. Prevent record/ID overlap and test-set tuning. Specify sampling with fixed seeds, duplicate checks, data provenance and how raw data is obtained without committing restricted data to GitHub.

Compare at least two baselines and the proposed approach:
1. Full-context synthesis;
2. Dense RAG;
3. Proposed retrieval-guided context arrangement + claim-level verifier + targeted re-retrieval;
4. An ablation of the proposed method with verification disabled, where compute permits.

Explain whether each baseline gets equivalent model access, generation settings, evidence budget and evaluation inputs. Recommend the smallest credible benchmark for a T4 and a CPU fallback.
```

**Save:** Dataset validation sheet, data-construction specification and comparator table.

## PROMPT 07 — Freeze the experiment blueprint before the final run

```text
Act as an experimental-design auditor. Freeze the protocol before inspecting any new test results.

Primary question: How do evidence position and corpus size affect evidence retention and source attribution, and does the proposed verification pipeline improve them relative to full-context and dense RAG?

Create an exact protocol including:
A. Inputs and outputs for every module;
B. Corpus construction and source-ID conventions;
C. Position manipulation (start, middle, end) and exact definition of where the target evidence is placed;
D. Corpus sizes (5, 15, 25), with the number of gold target documents and distractors stated;
E. Fixed seeds and number of independent clusters/runs;
F. Data partitions and leakage controls;
G. Frozen models, versions, prompts, generation parameters, retrieval top-k and NLI thresholds;
H. Token budget and maximum input/output lengths;
I. Hardware and time budget;
J. Logging and failure recovery;
K. Expected CSV schemas and figure definitions.

Compare these systems under the same task and evaluation conditions:
- Baseline 1: Full-context synthesis;
- Baseline 2: Dense-RAG synthesis;
- Proposed: retrieval ranking/context construction, claim-level evidence verification and targeted re-retrieval/re-synthesis;
- Ablation: same proposed retrieval/context construction with verification disabled.

Define metrics precisely:
- gold Evidence Recall@k / retrieval recall;
- evidence-nugget retention rate;
- qualifier retention rate (computed against exact gold annotations, not loose keyword presence alone);
- contradiction detection only where the benchmark has explicit gold contradiction labels;
- citation precision, citation recall and citation F1;
- claim support/entailment rate, with NLI model labels verified;
- latency per run and, if available, token count or memory use.

State which metrics are primary and secondary. Specify manual annotation/audit of a sample of generated claims and an inter-annotator agreement plan if more than one annotator is available.

Plan deeper analysis: position × corpus-size interaction, failure-case analysis, and verification ablation. Define the statistical summary (mean, standard deviation and/or bootstrap confidence interval) and do not claim significance unless a valid test and assumptions are justified.

Compute the expected run count. For the current three systems, 15 clusters × 3 corpus sizes × 3 positions × 3 systems = 405 runs. If the ablation is included as a fourth system for every cell, the total becomes 540. Also define a 5-cluster pilot (135 runs for three systems) before the full run. Make resource-saving rules explicit without changing the frozen test protocol silently.

Return an experiment flowchart, preregistration-style protocol, CSV schema, pre-run checklist and protocol-change log template. Do not predict outcomes.
```

**Save:** Frozen `experiment_protocol.md`, flowchart, metric formulas and pre-run checklist.

---

# Phase III — Coding, baseline verification and debugging

## PROMPT 08 — Verify Colab environment, model loading and reproducibility

```text
Act as a research software engineer. Audit and configure the actual Google Colab notebook for the frozen protocol.

Target environment: Google Colab, preferably Tesla T4 GPU. The earlier saved Gate G2 output reported a Tesla T4 with approximately 14.56 GB VRAM, but the later Gate G3 saved output reported CPU. Treat this as a runtime difference to diagnose, not as a contradiction to hide.

Inspect the notebook's install commands and versions. Provide a minimal environment cell that prints:
- Python, PyTorch, Transformers, Datasets, SentenceTransformers and Accelerate versions;
- CUDA availability, GPU name and free/total memory;
- model IDs and tokenizer/model configuration;
- device map and actual parameter placement;
- random seeds;
- output directory and disk space.

Check compatibility using actual imports and small test calls. Do not claim a package/model works until a real check passes. Avoid changing major library versions without documenting why.

Create an environment test that loads the dataset and each model separately, executes one short generation, one embedding/retrieval call and one NLI call, then releases memory safely. Print PASS/FAIL with a diagnostic message for each test.

Investigate why a saved run used CPU while the earlier run used CUDA. Explain how to select Runtime → Change runtime type → T4 GPU, verify `torch.cuda.is_available()`, and rerun all dependent cells in order.

Set deterministic seeds for Python, NumPy and PyTorch. Record that exact reproducibility may still vary by hardware/library kernels. Do not put Hugging Face tokens or secrets in the notebook or GitHub.

Stop after the environment test. Return the exact code cell and ask me to share its actual output before proceeding to the full experiment.
```

**Save:** Environment verification log and reproducible setup cell.

## PROMPT 09 — Implement or repair the pipeline one module at a time

```text
Act as my AI pair programmer and research software engineer. Work in small, reviewable code changes against my uploaded notebook. Do not rewrite unrelated cells.

Implement the frozen pipeline in this order:
1. Dataset loading, provenance and schema validation;
2. Controlled cluster construction with correct source IDs and gold evidence annotations;
3. Position/corpus-size manipulation;
4. Full-context baseline;
5. Dense-RAG baseline with explicit retrieval scores and retrieved document IDs;
6. Proposed evidence-aware context construction;
7. Atomic claim extraction or claim segmentation;
8. Claim-to-source mapping and NLI verification;
9. Targeted re-retrieval/re-synthesis for claims that lack support;
10. Evaluation and result export.

For each module use this cycle:
A. State the module's scientific purpose;
B. State the input/output schema and assumptions;
C. Identify the relevant existing code;
D. Provide the smallest code patch;
E. Explain every non-obvious line;
F. Add unit tests and edge cases;
G. Tell me exactly which cell to run and what output to capture;
H. Wait for the actual output before moving to the next module.

Mandatory corrections:
- Remove fictional author/year attributions and invented numeric claims from any cell that could be mistaken for real literature evidence.
- If synthetic qualifiers/contradictions are retained, label them `SYNTHETIC_CONTROLLED_TEST` and keep their gold labels and source IDs separate from PubMedQA provenance.
- Do not pair unrelated PubMedQA `yes` and `no` records and call them contradictory studies.
- Keep authentic PubMedQA fields unchanged in the natural-data track.
- Replace arbitrary keywords from `long_answer` with well-defined gold evidence units and transparent scoring; if an automatic proxy remains, name it as a proxy.
- Verify `nli_verifier.config.id2label` or the model's documented label mapping before interpreting class indices. Do not assume index 0/1/2 means contradiction/entailment/neutral.
- Check that the NLI premise is the cited source evidence and the hypothesis is one atomic generated claim, not the entire synthesis paragraph.
- Handle invalid/missing citation IDs, uncited claims, multiple citations, neutral NLI results and empty retrieved sets.
- Preserve original document IDs after ranking/reordering; never let displayed citation IDs silently point to different source documents.
- Log retrieval time, generation time, verification time and total latency separately where feasible.

Use type hints, small functions, informative errors and checkpointed intermediate outputs. Do not say “tested” until I provide actual execution output or you have genuinely executed the code in the available environment.
```

**Save:** Module-by-module implementation, tests and prompt-to-code record.

## PROMPT 10 — Validate the baselines and audit the current Gate G2 results

```text
Act as an independent reproducibility reviewer. Audit the existing Gate G2 and all baseline implementations before accepting any proposed-method result.

The uploaded notebook's saved Gate G2 cell reports progress through 135/135 runs for 5 clusters × 3 corpus sizes × 3 positions × 3 systems. It also reports a Tesla T4 in that cell. The saved Gate G3 cell is configured for 405 runs but its visible saved output does not show the 405-run progress or final result table. Verify these facts from the current notebook and any newly supplied CSV/log files; do not infer missing outputs.

Audit:
- exact source data and data-generation logic;
- positive/negative pair construction;
- whether target qualifier and contradiction statements are authentic or synthetic;
- corpus size and position construction;
- retrieval top-k and relevance labels;
- model and tokenizer versions;
- generation settings and prompt templates;
- citation-ID stability;
- NLI label mapping and threshold;
- metric formulas and aggregation;
- device, timing and memory logging;
- missing, duplicated or failed runs.

For every component label it VERIFIED, UNCERTAIN or NEEDS CORRECTION. Explain whether the 135-run Gate G2 can be retained as a pilot or must be rerun after correcting data/metrics. Do not silently carry invalid results into the final paper.

Create tests for:
1. exact corpus size;
2. exactly one gold location per controlled nugget;
3. position correctness;
4. source-ID/citation alignment after reordering;
5. retrieval gold-hit calculation;
6. NLI label mapping;
7. uncited-claim handling;
8. empty/malformed outputs;
9. run-count completeness;
10. CSV row uniqueness.

Compare results with published baselines only if the original evaluation protocol is sufficiently equivalent. Otherwise describe the results as an in-project controlled comparison, not an exact reproduction.

Return a baseline validation report, a list of corrections, rerun decision and the exact cells that need execution.
```

**Save:** `baseline_validation.md`, test report and rerun decision.

## PROMPT 11 — Debug a failure using the actual traceback or output

```text
Act as a careful debugging partner. Diagnose only the failure evidenced by the supplied code and actual output.

Environment: Google Colab; possible Tesla T4 GPU or CPU fallback. Models may include Qwen2.5-3B-Instruct, BGE small embeddings and DeBERTa NLI.

I will provide:
- the failing cell/function;
- the full traceback or exact unexpected output;
- the preceding environment diagnostics;
- the expected behavior;
- the last successful checkpoint.

Your response must:
1. Restate the observed failure precisely;
2. Separate confirmed facts from possible causes;
3. Identify the smallest safe correction;
4. Explain why the correction addresses the observed failure;
5. Provide a minimal patch, not an unrelated rewrite;
6. Provide a verification test and one edge case;
7. State whether earlier results are invalidated and must be rerun;
8. Add the case to `ai_assistance_log.md` with the prompt, AI suggestion, human decision, test and actual outcome.

Check likely failure categories only when supported by evidence: package incompatibility, CUDA/CPU device mismatch, VRAM exhaustion, `device_map`/offloading, model-label index assumptions, tokenizer padding, tensor device mismatch, malformed document IDs, bad citation parsing, missing dataset fields, empty retrieval results, CSV/schema mismatch, incomplete loops or interrupted Colab sessions.

Do not invent a traceback, execution result, successful fix or metric. If information is missing, request the single most diagnostic item.
```

**Save:** Corrected code, debugging record and retest evidence.

---

# Phase IV — Contribution, evaluation and analysis

## PROMPT 12 — Choose the defensible contribution and design ablations

```text
Act as a critical research-methodology mentor. My baseline validation report, actual baseline outputs and frozen research question are attached.

Compare three contribution routes:
A. Reproduce a known position/long-context diagnostic and extend it with stage-wise evidence tracking;
B. Conduct a controlled comparison of full-context, dense RAG and the proposed evidence-aware pipeline;
C. Make one targeted modification to context construction or claim-level verification.

Recommend the smallest defensible route based on the evidence and the Colab compute budget. Do not call a routine comparison novel without a literature search.

The candidate proposed system is based on the review's Figure 2 and currently includes retrieval-guided ranking/context ordering, claim-level source verification and targeted re-retrieval/re-synthesis. Audit whether each feature is genuinely implemented as described.

Design ablations that isolate components:
- proposed system with verification enabled;
- identical retrieval/context construction with verification disabled;
- if justified, verification without edge-salience reordering;
- optional comparison of original order versus re-ranked order.

For every ablation specify the changed component, everything held constant, sample subset, metric and interpretation. Avoid an excessive factorial design that cannot be completed.

State the contribution in three parts:
1. Prior work already established;
2. What this implementation operationalizes or evaluates;
3. What the measured results may support, depending on actual outcomes.

Return the selected pathway, ablation table, compute estimate, threats to validity and a concise contribution statement that does not overclaim.
```

**Save:** Contribution decision and ablation plan.

## PROMPT 13 — Evaluate only genuine outputs and create figures/tables

```text
Act as a strict experimental evaluator. Use only the actual CSVs, JSON logs, notebook outputs, model logs and manually verified annotations I attach. Do not invent missing values or reconstruct a result table from memory.

Before calculating metrics, validate:
- all expected run IDs are present;
- no duplicate cluster × size × position × system combinations;
- correct corpus size and position labels;
- valid source IDs and gold labels;
- output text is present and not truncated;
- all systems use the frozen protocol;
- no test examples were used for prompt/threshold tuning;
- failed/interrupted runs are reported, not silently dropped;
- synthetic and natural-data tracks remain separate.

Recompute the metrics from raw predictions wherever possible. Define exact denominators and formulas. Report retrieval-stage metrics separately from generation-stage metrics. Report NLI model predictions as model-based proxies unless validated against human annotations.

Produce:
1. `results.csv` with one row per run;
2. `summary_by_system.csv`;
3. `summary_by_position.csv`;
4. `summary_by_corpus_size.csv`;
5. an ablation summary;
6. a failure-case CSV with source documents, generated claim, citations, gold label, NLI output and error category;
7. publication-quality PNG and PDF plots.

Required figures:
- evidence retention by position and system;
- evidence retention versus corpus size;
- citation precision/recall/F1 by system;
- faithfulness/support proxy by system, with limitations labelled;
- latency versus quality trade-off;
- ablation results.

Show mean and standard deviation or justified confidence intervals. If only one seed is available, disclose it and do not imply robust statistical generalization. Perform significance testing only if the design and assumptions justify it.

Audit plotting code against the exported CSV. Every table and figure value must be reproducible from a named file and script. Provide the exact command/cell used to regenerate each artifact.
```

**Save:** Traceable result files, plots, evaluation script and metric audit.

## PROMPT 14 — Interpret positive, negative and ambiguous results

```text
Act as a skeptical results analyst and co-author. Use only the frozen protocol and the actual verified output tables, figures, logs and failure cases supplied here.

For each research question:
1. State the direct numerical observation;
2. State whether the hypothesis is supported, contradicted or unresolved;
3. Compare all systems under equal conditions;
4. Analyze position and corpus-size effects;
5. Separate retrieval failures from context-position loss, distractor/length effects, synthesis errors and attribution/verification errors;
6. Explain quality-versus-latency trade-offs;
7. Analyze at least five representative failure cases if available;
8. Discuss variation across seeds/clusters and uncertainty;
9. Provide alternative explanations and threats to validity;
10. State what cannot be concluded.

Do not equate keyword presence with semantic retention. Do not equate NLI entailment with factual truth. Do not treat a synthetic contradiction test as evidence about real medical literature. Do not claim that a system “solves hallucination” or “eliminates positional bias.”

Explicitly discuss the implications of synthetic controlled data versus natural PubMedQA validation. If the natural track does not support cross-paper contradiction conclusions, say so.

Recommend no more than three follow-up experiments, ranked by expected information gain and cost. Finish with a Results and Discussion outline, limitations paragraph and a concise answer to the primary research question, proportional to the evidence.
```

**Save:** Evidence-based Results/Discussion outline and limitations.

---

# Phase V — Manuscript, peer review and reproducibility

## PROMPT 15 — Draft the implementation research paper in LaTeX

```text
Act as an academic research-paper co-author and technical editor. Draft an implementation-based research paper using only verified evidence supplied in:
- original literature review and audited primary references;
- approved research brief and gap matrix;
- frozen experiment protocol;
- corrected notebook/code;
- baseline validation report;
- actual CSV results, figures, logs and failure cases;
- dataset/model documentation.

Provisional title: “Diagnosing Position- and Length-Related Evidence Loss in Long-Document Synthesis with Claim-Level Provenance Verification.” Revise the title if the implemented work does not support it.

Use an appropriate IEEE/ACM-style structure unless the instructor specifies another format:
1. Abstract (write last);
2. Keywords;
3. Introduction and problem statement;
4. Related work;
5. Research gap and questions;
6. Methodology and system architecture;
7. Dataset and controlled benchmark construction;
8. Experimental setup and baselines;
9. Metrics;
10. Results;
11. Error/failure analysis and ablations;
12. Discussion;
13. Threats to validity and limitations;
14. Conclusion and future work;
15. References.

Explain the architecture as modules: corpus/data provenance → indexing/embeddings → retrieval/context construction → synthesis → atomic claim extraction → source-level NLI verification → targeted re-retrieval/re-synthesis → evaluation.

Be exact about synthetic content. Give a separate subsection explaining which evidence is synthetic, how it was generated, why it was used and what conclusions it permits. Never attribute generated qualifiers, p-values or fictional study names to PubMedQA or real authors. Report natural PubMedQA results separately if available.

Do not invent any result, standard deviation, p-value, benchmark property, license, model capability or citation. Use `[TO VERIFY]` for missing facts. Create BibTeX only from verified bibliographic metadata. Ensure all tables and figure references match the actual files.

First deliver a section outline plus an evidence checklist. Then draft sections in order, leaving the abstract until the end. Include reproducibility details: seed, versions, GPU/CPU, model IDs, prompts, corpus sizes, positions, run counts, top-k, thresholds and commands. Distinguish previous literature from this study's implementation contribution.
```

**Save:** LaTeX manuscript, `.bib`, figures and evidence checklist.

## PROMPT 16 — Conduct a demanding technical peer review

```text
Act as a skeptical reviewer for an empirical NLP/IR research paper. Review the complete draft PDF, LaTeX source, bibliography, notebook, experiment protocol and result files.

Assess:
- whether the research question follows from the original review;
- whether the claimed gap is supported by primary literature;
- whether the contribution is distinct from prior work;
- whether synthetic and natural data are distinguished;
- whether PubMedQA records are used correctly;
- whether any fictional citations or artificial medical claims remain;
- whether baselines are fair and comparable;
- whether retrieval, position, length and synthesis failures are separately measured;
- whether claim-level verification is actually claim-level;
- whether NLI label mapping and thresholds are justified;
- whether citation IDs remain aligned after reordering;
- whether the evaluation metrics measure the stated constructs;
- whether the run count, CSVs, plots and manuscript tables agree;
- whether uncertainty, seeds, ablations and failure cases are adequate;
- whether conclusions exceed the evidence;
- whether another student can reproduce the work.

Return MAJOR and MINOR concerns. For each concern provide section/page/code location, why it matters, severity, evidence and one concrete remedy. Include a table mapping every paper claim to the code/result file that supports it.

Specifically check that the notebook's Gate G3 run is not called completed unless its actual final outputs exist. Check whether the Gate G2 135-run results remain valid after any data or metric corrections.

Do not invent citations or hypothetical reviewer requirements. End with a prioritized revision checklist and an author-response table.
```

**Save:** Technical peer-review report and revision log.

## PROMPT 17 — Prepare a reproducible GitHub repository and README

```text
Act as a research software release engineer. Prepare a clean, reproducible repository using only the actual project files, executed commands, verified dependencies and real results.

Use this structure unless the existing repository has a justified alternative:

AITR_Lab02_Project/
  README.md
  requirements.txt
  configs/
    experiment.yaml
  data/
    README.md
  notebooks/
    implementation.ipynb
  src/
    data.py
    retrieval.py
    synthesis.py
    verification.py
    metrics.py
  scripts/
    run_pilot.py
    run_experiment.py
    evaluate.py
    make_figures.py
  results/
    README.md
    results.csv (only if genuinely generated)
    summary_metrics.csv (only if genuinely generated)
    figures/ (only genuine figures)
  paper/
    manuscript.tex
    references.bib
    figures/
  research_decisions.md
  gap_matrix.md
  experiment_protocol.md
  baseline_validation.md
  ai_assistance_log.md

The README must include:
- research question and scope;
- contribution and limitations;
- synthetic versus natural-data separation;
- dataset access instructions and citation;
- model IDs and model/dataset licensing notes to verify;
- environment setup;
- GPU/CPU notes;
- exact pilot and full-run commands;
- evaluation and figure-generation commands;
- expected output filenames;
- random seed and configuration details;
- how to reproduce one run;
- known limitations and unresolved checks;
- citation and acknowledgement information.

Do not upload raw data if terms prohibit it. Do not include tokens, secrets, private paths or another person's personal metadata. Mark untested commands `[TEST REQUIRED]`; do not claim that a clean clone was tested unless it was.

Audit notebook outputs and result files before placing them in `results/`. Include only genuine outputs generated by the corrected protocol. If the earlier 135-run pilot was generated by the flawed synthetic data construction, preserve it only as a clearly labelled pilot/diagnostic artifact or exclude it from the final empirical claims.

Return the complete README, requirements file, config example, repository audit checklist and exact list of files needed from me.
```

**Save:** Reproducible repository documentation and clean project tree.

---

# Phase VI — Human–AI accountability and demonstration

## PROMPT 18 — Create the authentic human–AI collaboration and research-integrity audit

```text
Act as a research-integrity auditor. Use my actual prompts, chat records that I provide, code diffs, notebook outputs, debugging notes, commits, experiment files and human decisions.

Create an AI-assistance audit with one row per significant AI-assisted task:
- date;
- research stage;
- tool/model used (only if known);
- exact prompt or prompt identifier;
- AI suggestion;
- human decision (accepted, modified or rejected);
- verification method;
- code/document/result changed;
- actual outcome;
- evidence file or commit.

Include authentic examples of:
1. AI suggestions that were accepted after testing;
2. AI suggestions that required correction;
3. AI-generated claims or code that were rejected because they lacked evidence;
4. debugging or validation decisions made by the researcher.

At minimum, discuss the need to distinguish the hand-written Week-1 toy examples and inserted synthetic qualifiers/contradictions from real published evidence. Explain how the researcher verified or corrected these issues. Only call this a real human correction if the supplied history or code changes demonstrate that it happened.

Explain what AI was used for: ideation, coding, debugging, metric design, analysis, figures and manuscript drafting, as applicable. Separate AI-generated suggestions from human-verified decisions. Do not invent prompts, chat history, errors, fixes, commits or outcomes.

Identify remaining risks: hallucinated citations, synthetic-data leakage into claims, over-reliance on keyword metrics, unverified NLI label mapping, unequal baselines, incomplete runs and test-set tuning.

Produce:
- `ai_assistance_log.md`;
- a one-page AI-use and research-integrity statement;
- a short reflection on what the researcher learned about AI-assisted coding;
- a verification checklist for viva questions.
```

**Save:** Authentic AI-provenance report and research-integrity statement.

---

# Optional final prompt — Prepare the five-minute demonstration and viva

Although the assignment specifies 18 prompts, use this optional final prompt to prepare the demonstration after all results and files are ready.

```text
Act as a demanding project evaluator conducting a five-minute research demonstration and technical viva. Use only my final corrected notebook, repository, paper and actual outputs.

Prepare a timed demonstration:
0:00–0:40 — research problem, literature gap and research question;
0:40–1:20 — dataset, synthetic/natural data distinction and experimental protocol;
1:20–2:20 — show the three systems and explain the proposed pipeline;
2:20–3:20 — run one small example and trace a source citation through retrieval, synthesis and verification;
3:20–4:20 — show actual result tables, one plot, one failure case and the ablation;
4:20–5:00 — contribution, limitations and conclusion.

Explain every important function in plain language, including data loading, cluster construction, position manipulation, embeddings, retrieval, context ordering, generation, citation parsing, NLI verification, re-retrieval, scoring and CSV/figure export.

Generate at least 25 likely viva questions with technically correct answers. Include questions about:
- what context-window degradation means;
- why nominal context length differs from effective evidence use;
- why full-context and RAG are baselines;
- why position and corpus size are independent variables;
- how the gold evidence is defined;
- why synthetic data are used and what they cannot prove;
- what NLI entailment means and does not mean;
- why citation precision and recall matter;
- what the ablation establishes;
- how latency is measured;
- how leakage and fabricated citations were prevented;
- what negative results mean;
- how to rerun the experiment.

Do not create a script that claims unperformed results. If a result or file is missing, identify it as missing rather than filling it in.
```

---

## Recommended order of execution

| Order | Prompt | Gate / output | Do not proceed until |
|---:|---|---|---|
| 1 | 01 | Workspace and notebook audit | Files and unknowns are listed |
| 2 | 02 | Evidence-based gaps | Original review/source claims are separated from inference |
| 3 | 03 | Literature-to-code map | One feasible study is selected |
| 4 | 04 | Feasibility and source audit | Data, model and literature risks are documented |
| 5 | 05 | Gate G1 research brief | Research question and metrics are measurable |
| 6 | 06 | Gate G2 dataset/comparator sheet | Synthetic and natural data are separated |
| 7 | 07 | Frozen protocol | Run count, metrics and controls are fixed |
| 8 | 08 | Environment test | Actual environment output passes |
| 9 | 09 | Working modules | Each module passes its test |
| 10 | 10 | Baseline validation | Baselines and metrics are audited |
| 11 | 11 | Debugging record | Actual failure is fixed and retested |
| 12 | 12 | Contribution and ablation plan | Proposed contribution is isolated |
| 13 | 13 | Gate G3 evidence | CSVs and figures derive from genuine runs |
| 14 | 14 | Results and discussion | Claims match measured evidence |
| 15 | 15 | Paper and LaTeX | Every result/citation is traceable |
| 16 | 16 | Peer review | Major concerns are resolved or disclosed |
| 17 | 17 | Repository | Another student can follow tested steps |
| 18 | 18 | AI-use audit | Human decisions and AI contributions are authentic |
| Optional | Demo | Gate G4 presentation | Notebook, results and paper agree |

## Files to attach when running the prompt sequence

Always attach the most relevant current versions, not only old screenshots:

1. Original literature-review PDF (the actual review manuscript, not only the assignment sheet or gap-matrix summary).
2. Prompt 01–02 gap-matrix PDF.
3. Latest `.ipynb` notebook.
4. Any genuine result CSV/JSON/logs and figures.
5. `research_decisions.md`, `experiment_protocol.md` and `baseline_validation.md` as they are created.
6. Verified bibliography or DOI/arXiv metadata for cited primary papers.

## Final research-integrity rule

A notebook cell that contains code is not proof that the cell completed. A configured run count is not a completed run count. A model-generated citation is not proof of a source. A keyword match or NLI score is not proof of factual truth. Use only genuine, traceable executions and clearly disclose the scope and limitations of synthetic tests.
