"""Controlled factorial benchmark for context-window degradation.

The stimulus construction mirrors the assignment protocol: a PubMedQA-derived
positive target (Study A), a labelled synthetic contradictory replication
(Study B), and topic-matched distractors. The negative replication is a controlled
counterfactual stimulus, NOT a claim about a real PubMedQA publication.

Run:
    python scripts/run_factorial_experiment.py --clusters 5
    python scripts/run_factorial_experiment.py --clusters 15
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "exp_config.yaml").read_text(encoding="utf-8"))
RESULTS = ROOT / "results"
CITATION_GROUP_RE = re.compile(r"\[([^\]]*)\]", re.IGNORECASE)
DOC_ID_RE = re.compile(r"\bDoc\s*(\d+)\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_passage(row: dict[str, Any]) -> str:
    context = row.get("context") or {}
    if isinstance(context, dict):
        pieces = context.get("contexts", [])
    elif isinstance(context, list):
        pieces = context
    else:
        pieces = [str(context)]
    if isinstance(pieces, str):
        pieces = [pieces]
    return " ".join(str(x).strip() for x in pieces if x).strip()


def load_pubmedqa() -> dict[str, list[dict[str, Any]]]:
    """Load the official labelled train split; raw records are not stored in Git."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    groups: dict[str, list[dict[str, Any]]] = {"yes": [], "no": [], "maybe": []}
    for idx, row in enumerate(ds):
        label = str(row.get("final_decision", "")).lower().strip()
        passage = extract_passage(row)
        question = str(row.get("question", "")).strip()
        if label in groups and question and len(passage) >= 80:
            groups[label].append({
                "source_id": str(idx),
                "question": question,
                "passage": passage,
                "long_answer": str(row.get("long_answer", "")).strip(),
                "decision": label,
            })
    if not groups["yes"] or not groups["no"] or len(groups["maybe"]) < 23:
        raise RuntimeError(
            "PubMedQA split did not provide enough yes/no/maybe records. "
            f"Counts: { {k: len(v) for k, v in groups.items()} }"
        )
    return groups


def content_terms(text: str) -> set[str]:
    stop = {"which", "what", "when", "where", "does", "with", "from", "this",
            "that", "were", "have", "been", "into", "among", "after", "before"}
    return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text) if w.lower() not in stop}


def make_stimulus_clusters(groups: dict[str, list[dict[str, Any]]],
                           n_clusters: int, seed: int) -> list[dict[str, Any]]:
    """Build controlled A/B evidence pairs plus distinct topic-matched distractors."""
    rng = random.Random(seed)
    positives, negatives, maybe_records = groups["yes"], groups["no"], groups["maybe"]
    clusters = []
    max_distractors = max(CONFIG["dataset"]["corpus_sizes_k"]) - 2

    for idx in range(n_clusters):
        pos = positives[idx % len(positives)]
        neg = negatives[idx % len(negatives)]
        question = pos["question"]
        ctx_parts = [p.strip() for p in pos["passage"].split(". ") if p.strip()]
        intro_part = (ctx_parts[0] if ctx_parts else pos["passage"])[:380]
        result_part = " ".join(ctx_parts[1:])[:420] or pos["long_answer"][:420]

        # This controlled qualifier is deliberately inserted in the middle of
        # Study A so its retention can be tested; it is a benchmark stimulus.
        qualifier_phrase = "subgroup under 65 years without renal impairment"
        positive_text = (
            f"Study A (PubMedQA source record {pos['source_id']}): Background and Design: "
            f"{intro_part}. Methodological Qualifier: Stratified analysis revealed that "
            f"the treatment benefit was observed only in the {qualifier_phrase}. "
            f"Primary Outcome: {result_part or pos['long_answer']}"
        )

        neg_body = neg["passage"][:550]
        # Assignment-specified synthetic counterfactual. Do not interpret the
        # inserted result/p-value as a statement made by the source paper.
        negative_text = (
            f"Study B (synthetic contradictory replication; source record "
            f"{neg['source_id']} used only as background text): {neg_body}. "
            f"CONTROLLED CONTRADICTION STIMULUS: Contrary to the research question "
            f"'{question}', this constructed replication found no statistically "
            f"significant benefit over standard control (p = 0.43) and reported "
            f"a conflicting outcome. This sentence is a benchmark counterfactual, "
            f"not an extracted PubMedQA finding."
        )

        query_terms = content_terms(question)
        scored = []
        for record in maybe_records:
            if record["source_id"] in {pos["source_id"], neg["source_id"]}:
                continue
            terms = content_terms(record["question"] + " " + record["passage"][:300])
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            scored.append((overlap, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        candidate_pool = [r for _, r in scored[:max(60, max_distractors * 3)]]
        if len(candidate_pool) < max_distractors:
            candidate_pool = [r for r in maybe_records
                              if r["source_id"] not in {pos["source_id"], neg["source_id"]}]
        selected = rng.sample(candidate_pool, k=max_distractors)

        clusters.append({
            "cluster_id": f"C{idx + 1:02d}",
            "question": question,
            "paper_pos": positive_text,
            "paper_neg": negative_text,
            "positive_source_id": pos["source_id"],
            "negative_source_id": neg["source_id"],
            "distractors": [
                {
                    "text": f"Topical distractor (PubMedQA record {r['source_id']}): "
                            f"{r['passage'][:1100]}",
                    "source_id": r["source_id"],
                }
                for r in selected
            ],
            "h1_keywords": ["under 65", "renal impairment", "subgroup", "only in"],
            "h4_keywords": ["contrary", "contradictory", "no statistically significant"],
            "headline_keywords": sorted(content_terms(pos["long_answer"]))[:20],
        })
    return clusters


def assemble_prompt_docs(contexts, max_tokens=3000):
    """
    Assembles and filters prompt context documents, ensuring all required 
    metadata keys (like 'role') are present to prevent KeyErrors.
    """
    assembled = []
    total_tokens = 0

    for idx, ctx in enumerate(contexts):
        # Ensure ctx is a dictionary and safely handle missing keys
        if isinstance(ctx, str):
            doc_item = {
                "id": f"doc_{idx}",
                "text": ctx,
                "role": "context",  # Safe default role
                "token_count": len(ctx.split())  # Approximate if tokenizer not passed
            }
        elif isinstance(ctx, dict):
            # Fallback for role if missing from incoming dict
            role = ctx.get("role") or ctx.get("type") or "context"
            text = ctx.get("text") or ctx.get("content", "")
            doc_id = ctx.get("id", f"doc_{idx}")
            
            doc_item = {
                "id": doc_id,
                "text": text,
                "role": role,  # Guaranteed 'role' key exists
                "token_count": ctx.get("token_count", len(text.split()))
            }
        else:
            continue

        # Optional token budget check
        if total_tokens + doc_item["token_count"] > max_tokens:
            break

        assembled.append(doc_item)
        total_tokens += doc_item["token_count"]

    return assembled

def load_models() -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = CONFIG["models"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["generator_llm"])
    generator = AutoModelForCausalLM.from_pretrained(
        model_cfg["generator_llm"],
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    generator.eval()
    return {
        "device": device,
        "tokenizer": tokenizer,
        "generator": generator,
        "retriever": SentenceTransformer(model_cfg["dense_retriever"], device=device),
        "verifier": CrossEncoder(model_cfg["nli_verifier"], device=device),
    }


def nli_label_indices(verifier: CrossEncoder) -> tuple[int, int]:
    id2label = getattr(verifier.model.config, "id2label", {}) or {}
    labels = {int(i): str(label).lower() for i, label in id2label.items()}
    entailment = next((i for i, label in labels.items() if "entail" in label), None)
    contradiction = next((i for i, label in labels.items() if "contrad" in label), None)
    # The published cross-encoder/nli-deberta-v3-small head uses
    # LABEL_0=contradiction, LABEL_1=entailment, LABEL_2=neutral.
    if entailment is None:
        entailment = 1
    if contradiction is None:
        contradiction = 0
    return entailment, contradiction


def make_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(f"[Doc {d['doc_id']}] {d['text']}" for d in docs)
    return (
        "You are an academic literature synthesis assistant. Answer the research "
        "question using only the supplied documents. Preserve subgroup, population, "
        "dose, duration, and uncertainty qualifiers. Explicitly report conflicting "
        "findings rather than smoothing them into consensus. Cite each factual "
        "sentence with [Doc N]. Do not invent sources. The evidence includes a "
        "controlled synthetic counterfactual; treat it as an experimental stimulus.\n\n"
        f"Documents:\n{evidence}\n\nResearch question: {question}\n\n"
        "Write a concise evidence-grounded synthesis with citations."
    )


def generate(models: dict[str, Any], question: str, docs: list[dict[str, Any]],
             max_new_tokens: int) -> str:
    tokenizer = models["tokenizer"]
    model = models["generator"]
    prompt = make_prompt(question, docs)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(formatted, return_tensors="pt", truncation=True,
                        max_length=24000)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def rank_docs(models: dict[str, Any], question: str,
              docs: list[dict[str, Any]]) -> tuple[list[int], np.ndarray]:
    retriever = models["retriever"]
    query_vec = retriever.encode([question], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0]
    doc_vecs = retriever.encode(
        [d["text"] for d in docs], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False
    )
    scores = doc_vecs @ query_vec
    ranked = np.argsort(scores)[::-1].astype(int).tolist()
    return ranked, scores


def alternating_edge_order(ranked: list[int]) -> list[int]:
    """Place ranked evidence alternately at prompt start and end (Figure 2 Stage A)."""
    left: list[int] = []
    right: list[int] = []
    for rank, index in enumerate(ranked):
        if rank % 2 == 0:
            left.append(index)
        else:
            right.insert(0, index)
    return left + right


def run_system(models: dict[str, Any], system: str, question: str,
               docs: list[dict[str, Any]], max_new_tokens: int
               ) -> tuple[str, float, list[dict[str, Any]]]:
    """Return the exact evidence documents supplied to the final answer generator."""
    start = time.perf_counter()
    if system == "Baseline_1_FullContext":
        evidence_docs = docs
        answer = generate(models, question, evidence_docs, max_new_tokens)
    elif system == "Baseline_2_DenseRAG":
        ranked, _ = rank_docs(models, question, docs)
        evidence_docs = [docs[i] for i in ranked[:min(4, len(ranked))]]
        answer = generate(models, question, evidence_docs, max_new_tokens)
    elif system == "Proposed_Fig2_Verified":
        ranked, _ = rank_docs(models, question, docs)
        reordered = [docs[i] for i in alternating_edge_order(ranked)]
        draft = generate(models, question, reordered, max_new_tokens)

        # Stage B checks citation coverage and contradiction, not whole-draft entailment.
        # A multi-document synthesis need not be entailed by each individual paper.
        if verify_draft_citations(models["verifier"], draft, reordered):
            answer = draft
            evidence_docs = reordered
        else:
            focused_docs = [docs[i] for i in ranked[:min(2, len(ranked))]]
            answer = generate(models, question, focused_docs, max_new_tokens)
            evidence_docs = focused_docs
    else:
        raise ValueError(f"Unknown system: {system}")
    return answer, time.perf_counter() - start, evidence_docs


def citation_ids(sentence: str) -> list[int]:
    """Extract every document ID from [Doc 1], [Doc 1, Doc 2], or adjacent tags."""
    found: list[int] = []
    for match in CITATION_GROUP_RE.finditer(sentence):
        group = match.group(1)
        if re.search(r"\bDoc\b", group, flags=re.IGNORECASE):
            found.extend(int(n) for n in DOC_ID_RE.findall(group))
    return sorted(set(found))


def strip_citations(sentence: str) -> str:
    """Remove document citation groups before sending a claim to the NLI model."""
    return CITATION_GROUP_RE.sub(
        lambda match: "" if re.search(r"\bDoc\s*\d+", match.group(1), re.IGNORECASE)
        else match.group(0),
        sentence,
    ).strip()


def verify_draft_citations(verifier: CrossEncoder, answer: str,
                           docs: list[dict[str, Any]],
                           contradiction_threshold: float = 0.25) -> bool:
    """Require valid citations on every sentence and reject strong cited contradictions."""
    doc_map = {doc["doc_id"]: doc for doc in docs}
    sentences = [s.strip() for s in SENTENCE_RE.split(answer) if s.strip()]
    if not sentences:
        return False

    entail_idx, contradiction_idx = nli_label_indices(verifier)
    for sentence in sentences:
        ids = citation_ids(sentence)
        if not ids or any(doc_id not in doc_map for doc_id in ids):
            return False
        claim = strip_citations(sentence)
        if not claim:
            return False
        pairs = [(doc_map[doc_id]["text"], claim) for doc_id in ids]
        probabilities = np.atleast_2d(verifier.predict(pairs, apply_softmax=True))
        if any(float(row[contradiction_idx]) >= contradiction_threshold
               for row in probabilities):
            return False
    return True


def h3_citation_faithfulness(models: dict[str, Any], answer: str,
                             docs: list[dict[str, Any]]) -> float:
    """Fraction of cited sentences supported by at least one cited document via NLI."""
    doc_map = {doc["doc_id"]: doc for doc in docs}
    sentences = [s.strip() for s in SENTENCE_RE.split(answer) if s.strip()]
    tested = supported = 0
    entail_idx, contradiction_idx = nli_label_indices(models["verifier"])
    for sentence in sentences:
        ids = citation_ids(sentence)
        if not ids:
            continue
        tested += 1
        cited_docs = [doc_map[i] for i in ids if i in doc_map]
        if len(cited_docs) != len(ids):
            continue
        claim = strip_citations(sentence)
        if not claim:
            continue
        pairs = [(doc["text"], claim) for doc in cited_docs]
        scores = np.atleast_2d(models["verifier"].predict(pairs, apply_softmax=True))
        if any(
            float(row[entail_idx]) > float(row[contradiction_idx])
            and float(row[contradiction_idx]) < 0.20
            for row in scores
        ):
            supported += 1
    return round(100.0 * supported / tested, 2) if tested else 0.0


def score_output(models: dict[str, Any], answer: str,
                 docs: list[dict[str, Any]], cluster: dict[str, Any]) -> dict[str, float]:
    low = answer.lower()
    headline_terms = cluster["headline_keywords"]
    headline_recall = (
        100.0 * sum(1 for term in headline_terms if term in low) / len(headline_terms)
        if headline_terms else 0.0
    )
    h1 = 100.0 if any(term in low for term in cluster["h1_keywords"]) else 0.0
    h4 = 100.0 if any(term in low for term in cluster["h4_keywords"]) else 0.0
    h3 = h3_citation_faithfulness(models, answer, docs)
    return {
        "headline_recall_pct": round(headline_recall, 2),
        "h1_qualifier_recall_pct": h1,
        "h4_contradiction_pct": h4,
        "h3_nli_faithfulness_pct": h3,
    }


def run_experiment(n_clusters: int, seed: int, max_new_tokens: int) -> pd.DataFrame:
    set_seed(seed)
    groups = load_pubmedqa()
    print("PubMedQA records:", {name: len(items) for name, items in groups.items()})
    clusters = make_stimulus_clusters(groups, n_clusters, seed)
    models = load_models()
    positions = CONFIG["dataset"]["target_positions"]
    corpus_sizes = CONFIG["dataset"]["corpus_sizes_k"]
    systems = CONFIG["systems_compared"]
    total_runs = n_clusters * len(corpus_sizes) * len(positions) * len(systems)
    rows: list[dict[str, Any]] = []
    completed = 0

    for cluster in clusters:
        for k in corpus_sizes:
            for position in positions:
                docs = assemble_prompt_docs(cluster, int(k), position)
                for system in systems:
                    answer, latency, evidence_docs = run_system(
                        models, system, cluster["question"], docs, max_new_tokens
                    )
                    # Evaluate citations only against evidence actually shown to this system.
                    metrics = score_output(models, answer, evidence_docs, cluster)
                    rows.append({
                        "cluster_id": cluster["cluster_id"],
                        "k_docs": int(k),
                        "position": position,
                        "system": system,
                        "latency_sec": round(latency, 3),
                        **metrics,
                        "generated_text": answer,
                        "document_manifest": json.dumps([
                            {"doc_id": d["doc_id"], "role": d["role"],
                             "source_id": d["source_id"]} for d in evidence_docs
                        ]),
                    })
                    completed += 1
                    print(f"Progress: {completed}/{total_runs} | {cluster['cluster_id']} "
                          f"| k={k} | {position} | {system} | {latency:.2f}s")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=int(CONFIG["dataset"]["pilot_clusters"]))
    parser.add_argument("--seed", type=int, default=int(CONFIG["seed"]))
    parser.add_argument("--max-new-tokens", type=int, default=int(CONFIG["decoding"]["max_new_tokens"]))
    args = parser.parse_args()
    if args.clusters < 1:
        parser.error("--clusters must be positive")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = run_experiment(args.clusters, args.seed, args.max_new_tokens)
    raw_path = RESULTS / "experiment_results.csv"
    summary_path = RESULTS / "summary_metrics.csv"
    df.to_csv(raw_path, index=False)

    metrics = ["headline_recall_pct", "h1_qualifier_recall_pct",
               "h4_contradiction_pct", "h3_nli_faithfulness_pct", "latency_sec"]
    summary = (df.groupby(["system", "k_docs", "position"])[metrics]
                 .agg(["mean", "std"]).reset_index())
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(summary_path, index=False)

    run_info = {
        "seed": args.seed,
        "clusters": args.clusters,
        "runs_expected": args.clusters * 3 * 3 * 3,
        "runs_written": len(df),
        "max_new_tokens": args.max_new_tokens,
        "dataset": CONFIG["dataset"]["source"],
        "systems": CONFIG["systems_compared"],
        "stimulus_note": "Study B includes an explicitly labelled synthetic contradiction.",
    }
    (RESULTS / "run_metadata.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(df)} measured runs to {raw_path}")
    print(f"Saved grouped mean/std to {summary_path}")


if __name__ == "__main__":
    main()
