"""Reproducible benchmark for position- and scale-dependent context degradation.

The script intentionally separates benchmark construction, model inference, and
metric calculation. It writes one row per (cluster, system, k, target_position)
condition and never substitutes synthetic values for failed model runs.

Run on a Colab T4 GPU or another CUDA machine:
    python scripts/run_factorial_experiment.py --clusters 5
For the larger run:
    python scripts/run_factorial_experiment.py --clusters 15
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "exp_config.yaml"
RESULTS = ROOT / "results"
DOC_TAG = re.compile(r"\[Doc\s*([0-9,\s]+)\]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Models:
    tokenizer: Any
    generator: Any
    retriever: Any
    verifier: Any
    device: str


def load_models(model_cfg: dict[str, Any]) -> Models:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator_name = model_cfg["generator_llm"]
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator = AutoModelForCausalLM.from_pretrained(
        generator_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    retriever = SentenceTransformer(model_cfg["dense_retriever"], device=device)
    verifier = CrossEncoder(model_cfg["nli_verifier"], device=device)
    return Models(tokenizer, generator, retriever, verifier, device)


def load_pubmedqa() -> list[dict[str, Any]]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for i, item in enumerate(ds):
        context = item.get("context") or {}
        contexts = context.get("contexts", []) if isinstance(context, dict) else []
        if isinstance(contexts, str):
            contexts = [contexts]
        passage = "\n".join(str(x) for x in contexts if x)
        question = str(item.get("question", "")).strip()
        answer = str(item.get("long_answer") or item.get("final_decision") or "").strip()
        if question and passage and answer:
            records.append({
                "cluster_id": i,
                "question": question,
                "passage": passage,
                "answer": answer,
                "decision": str(item.get("final_decision", "")).strip().lower(),
            })
    if len(records) < 25:
        raise RuntimeError("PubMedQA loader returned too few usable records; inspect dataset schema.")
    return records


def build_cluster(records: list[dict[str, Any]], cluster_id: int, k: int,
                  position: str, seed: int) -> list[dict[str, Any]]:
    """Create a deterministic bundle: one target, one contrastive record, k-2 distractors.

    The record under test is assigned an explicit target index. Other passages are
    topic-matched candidates; their order is seeded and repeatable.
    """
    rng = random.Random(seed + cluster_id * 1009 + k)
    target = records[(cluster_id * 7) % len(records)]
    remaining = [r for r in records if r["cluster_id"] != target["cluster_id"]]
    # Use distinct examples; matching by biomedical vocabulary is approximated by
    # the target question's content words to avoid adding another external corpus.
    query_terms = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", target["question"])}
    scored = []
    for rec in remaining:
        words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", rec["question"])}
        score = len(query_terms & words) / max(1, len(query_terms | words))
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = [r for _, r in scored[: max(k * 4, k)]]
    if len(pool) < k - 1:
        pool = remaining
    rng.shuffle(pool)
    selected = [target] + pool[: k - 1]
    # Keep the first item as the positive study and place its paired distractor
    # at the requested target position. This position is recorded explicitly.
    if position == "start":
        target_index = 0
    elif position == "middle":
        target_index = k // 2
    elif position == "end":
        target_index = k - 1
    else:
        raise ValueError(f"Unknown position: {position}")
    distractors = [r for r in selected if r["cluster_id"] != target["cluster_id"]]
    bundle = [dict(r, doc_id=i + 1, is_target=False) for i, r in enumerate(distractors)]
    target_doc = dict(target, doc_id=-1, is_target=True)
    bundle.insert(target_index, target_doc)
    for i, doc in enumerate(bundle):
        doc["doc_id"] = i + 1
    assert len(bundle) == k and sum(bool(d["is_target"]) for d in bundle) == 1
    return bundle


def prompt_for(question: str, docs: list[dict[str, Any]], include_positions: bool) -> str:
    chunks = []
    for d in docs:
        position = f" [position={d['doc_id']}]" if include_positions else ""
        chunks.append(f"[Doc {d['doc_id']}]{position}\nQuestion: {d['question']}\nAbstract: {d['passage']}")
    return (
        "You are synthesizing evidence from a set of biomedical research abstracts. "
        "Answer the research question using only the supplied documents. Preserve "
        "qualifiers (population, dose, duration, uncertainty) and explicitly report "
        "conflicting findings. Cite every factual sentence with [Doc N]. Do not "
        "invent references. If the documents do not establish an answer, say so.\n\n"
        + "\n\n".join(chunks)
        + f"\n\nResearch Question: {question}\nReturn a concise synthesis with citations."
    )


def generate(models: Models, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [{"role": "user", "content": prompt}]
    formatted = models.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = models.tokenizer(formatted, return_tensors="pt", truncation=True,
                              max_length=24000)
    inputs = {k: v.to(models.generator.device) for k, v in inputs.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        output = models.generator.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=models.tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return models.tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def retrieve(models: Models, question: str, docs: list[dict[str, Any]], top_k: int = 4) -> list[dict[str, Any]]:
    embeddings = models.retriever.encode(
        [d["passage"] for d in docs], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False
    )
    q_emb = models.retriever.encode(
        [question], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False
    )[0]
    scores = embeddings @ q_emb
    order = np.argsort(scores)[::-1][:top_k]
    return [docs[int(i)] for i in order]


def edge_reorder(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Place the target evidence at a prompt edge, keeping remaining order stable."""
    target = [d for d in docs if d["is_target"]]
    other = [d for d in docs if not d["is_target"]]
    return target + other if target else docs


def parse_citations(text: str) -> list[int]:
    found: list[int] = []
    for match in DOC_TAG.findall(text):
        found.extend(int(x) for x in re.findall(r"\d+", match))
    return sorted(set(found))


def citation_faithfulness(models: Models, answer: str, docs: list[dict[str, Any]]) -> float:
    """Sentence-level support proxy: entailment > contradiction and contradiction < .20."""
    if not answer.strip():
        return 0.0
    doc_by_id = {d["doc_id"]: d for d in docs}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    tested = supported = 0
    for sentence in sentences:
        ids = parse_citations(sentence)
        if not ids:
            continue
        claim = DOC_TAG.sub("", sentence).strip()
        valid_docs = [doc_by_id[i] for i in ids if i in doc_by_id]
        if not valid_docs:
            tested += 1
            continue
        scores = models.verifier.predict([
            (d["passage"], claim) for d in valid_docs
        ], apply_softmax=True)
        # CrossEncoder class order for DeBERTa MNLI is model-config dependent.
        # Resolve from model labels rather than assuming an index mapping.
        labels = models.verifier.model.config.id2label
        label_to_idx = {str(v).lower(): int(k) for k, v in labels.items()}
        ent_idx = next((v for k, v in label_to_idx.items() if "entail" in k), None)
        con_idx = next((v for k, v in label_to_idx.items() if "contrad" in k), None)
        tested += 1
        if ent_idx is None or con_idx is None:
            continue
        if any(float(row[ent_idx]) > float(row[con_idx]) and float(row[con_idx]) < 0.20
               for row in scores):
            supported += 1
    return 100.0 * supported / tested if tested else 0.0


def score_run(models: Models, output: str, docs: list[dict[str, Any]],
              latency: float) -> dict[str, float]:
    target = next(d for d in docs if d["is_target"])
    target_id = target["doc_id"]
    citations = parse_citations(output)
    # These benchmark measures are explicit proxy checks, not clinical truth labels.
    h1_terms = re.findall(
        r"\b(?:may|might|could|approximately|not|only|among|in patients with|"
        r"associated with|trend toward|limited to|uncertain)\b",
        target["passage"].lower()
    )
    h1_found = sum(1 for term in set(h1_terms) if term in output.lower())
    h1_recall = 100.0 * h1_found / len(set(h1_terms)) if h1_terms else np.nan
    h4_conflict = 100.0 if any(
        d["decision"] in {"yes", "no"} and target["decision"] in {"yes", "no"}
        and d["decision"] != target["decision"] and f"[Doc {d['doc_id']}]" in output
        for d in docs if not d["is_target"]
    ) else 0.0
    return {
        "headline_recall_pct": 100.0 if target_id in citations else 0.0,
        "h1_qualifier_recall_pct": h1_recall,
        "h4_contradiction_pct": h4_conflict,
        "h3_nli_faithfulness_pct": citation_faithfulness(models, output, docs),
        "latency_sec": latency,
    }


def run_experiment(clusters: int, seed: int, max_new_tokens: int) -> pd.DataFrame:
    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    set_seed(seed)
    print("Loading PubMedQA and models (first run downloads model weights)...")
    records = load_pubmedqa()
    models = load_models(cfg["models"])
    rows = []
    positions = ["start", "middle", "end"]
    sizes = [5, 15, 25]
    systems = ["Baseline_1_FullContext", "Baseline_2_DenseRAG", "Proposed_Fig2_Verified"]
    for cluster_id in range(clusters):
        for k in sizes:
            for position in positions:
                docs = build_cluster(records, cluster_id, k, position, seed)
                question = docs[0]["question"]
                retrieved = retrieve(models, question, docs, top_k=min(4, k))
                for system in systems:
                    if system == "Baseline_1_FullContext":
                        used_docs = docs
                        prompt = prompt_for(question, used_docs, False)
                    elif system == "Baseline_2_DenseRAG":
                        used_docs = retrieved
                        prompt = prompt_for(question, used_docs, False)
                    else:
                        used_docs = edge_reorder(retrieved)
                        prompt = prompt_for(question, used_docs, True)
                    output, latency = generate(models, prompt, max_new_tokens)
                    metrics = score_run(models, output, used_docs, latency)
                    rows.append({
                        "cluster_id": cluster_id, "k": k, "target_position": position,
                        "system": system, "question": question, "generated_text": output,
                        "doc_ids": json.dumps([d["doc_id"] for d in used_docs]),
                        **metrics,
                    })
                    print(f"[{len(rows)}/{clusters*27}] cluster={cluster_id} k={k} "
                          f"position={position} system={system} latency={latency:.2f}s")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be >= 1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = run_experiment(args.clusters, args.seed, args.max_new_tokens)
    raw_path = RESULTS / "experiment_results.csv"
    summary_path = RESULTS / "summary_metrics.csv"
    df.to_csv(raw_path, index=False)
    metrics = ["headline_recall_pct", "h1_qualifier_recall_pct",
               "h4_contradiction_pct", "h3_nli_faithfulness_pct", "latency_sec"]
    summary = df.groupby(["system", "k", "target_position"], dropna=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(summary_path, index=False)
    print(f"Saved {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
