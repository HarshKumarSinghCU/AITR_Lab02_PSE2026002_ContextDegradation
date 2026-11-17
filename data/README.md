# Dataset and stimulus protocol

## Dataset source

- **Dataset:** `qiaojin/PubMedQA`
- **Configuration:** `pqa_labeled`
- **Split:** `train`
- **Source:** https://huggingface.co/datasets/qiaojin/PubMedQA

The raw dataset is downloaded at execution time through the Hugging Face `datasets` package. It is not stored in this repository.

## Controlled benchmark construction

The assignment's factorial benchmark requires a paired positive and contradictory target plus (k-2) topical distractors. The script selects source abstracts from the PubMedQA yes/no/maybe decision groups:

- Study A is built from a PubMedQA yes-record and includes the controlled H1 subgroup qualifier.
- Study B uses a PubMedQA no-record as background text and adds an explicit, labelled synthetic contradictory-replication sentence for the H4 test.
- Distractors are sampled deterministically from maybe-records, preferentially using overlap with the target question.

**Important:** the Study B contradiction sentence and its illustrative p-value are controlled synthetic content, not a quotation or result from the source record. Report it as a benchmark intervention, not as a published biomedical finding.

## Reproducibility

The run uses seed 42 by default. Cluster count, corpus sizes and target positions are frozen in `configs/exp_config.yaml`. The run output includes a per-document manifest so source-record identifiers and roles can be audited. No raw patient-level data are required or written by the scripts.
