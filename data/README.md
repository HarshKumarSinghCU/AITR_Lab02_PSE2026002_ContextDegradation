# Dataset Access and Protocol (`data/README.md`)

## 1. Official Data Source & Licensing
- **Dataset:** `qiaojin/PubMedQA` (`pqa_labeled` subset)
- **Source URL:** https://huggingface.co/datasets/qiaojin/PubMedQA
- **License:** MIT License (publicly accessible; no restricted patient or proprietary data)
- **Total Records:** 1,000 expert-annotated biomedical research QA instances (551 `yes`, 338 `no`, 109 `maybe`).

## 2. How the Dataset is Loaded (Zero Raw Data Stored in Repo)
In accordance with repository hygiene rules, raw dataset files are downloaded dynamically via the HuggingFace `datasets` library during script execution:
```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
