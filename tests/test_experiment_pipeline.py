"""Unit tests for deterministic factorial document assembly and citation parsing."""
import unittest

from src.experiment_pipeline import (
    assemble_prompt_docs,
    alternating_edge_order,
    citation_ids,
    strip_citations,
    verify_draft_citations,
)


def fixture_cluster():
    return {
        "cluster_id": "TEST",
        "question": "Does intervention A improve outcome B?",
        "paper_pos": "Study A positive target with a controlled qualifier.",
        "paper_neg": "Study B synthetic contradictory target.",
        "positive_source_id": "yes-0",
        "negative_source_id": "no-0",
        "distractors": [
            {"text": f"Topical distractor {i}", "source_id": f"maybe-{i}"}
            for i in range(23)
        ],
    }


class FactorialAssemblyTests(unittest.TestCase):
    def test_count_and_pair_positions_for_all_conditions(self):
        cluster = fixture_cluster()
        for k in (5, 15, 25):
            for position in ("start", "middle", "end"):
                with self.subTest(k=k, position=position):
                    docs = assemble_prompt_docs(cluster, k, position)
                    self.assertEqual(len(docs), k)
                    positive = next(i for i, d in enumerate(docs)
                                    if d["role"] == "positive_target")
                    negative = next(i for i, d in enumerate(docs)
                                    if d["role"] == "contradictory_target")
                    self.assertEqual(negative, positive + 1)
                    if position == "start":
                        self.assertEqual(positive, 0)
                    elif position == "end":
                        self.assertEqual(positive, k - 2)
                    else:
                        self.assertEqual(positive, (k - 2) // 2)

    def test_document_ids_are_unique_and_sequential(self):
        docs = assemble_prompt_docs(fixture_cluster(), 15, "middle")
        self.assertEqual([d["doc_id"] for d in docs], list(range(1, 16)))


class DummyNLIConfig:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class DummyNLIModel:
    config = DummyNLIConfig()


class DummyVerifier:
    model = DummyNLIModel()

    def __init__(self, contradiction=0.05):
        self.contradiction = contradiction

    def predict(self, pairs, apply_softmax=True):
        return [[self.contradiction, 0.85, 0.10] for _ in pairs]


class CitationAndOrderingTests(unittest.TestCase):
    def test_comma_separated_citations_are_parsed(self):
        self.assertEqual(citation_ids("Claim [Doc 1, Doc 2] with [Doc 4]."),
                         [1, 2, 4])

    def test_repeated_citations_are_deduplicated(self):
        self.assertEqual(citation_ids("[Doc 2] and [Doc 2, Doc 3]"), [2, 3])

    def test_adjacent_citation_tags_are_parsed(self):
        self.assertEqual(citation_ids("Claim [Doc 1][Doc 3]."), [1, 3])

    def test_citation_tags_are_removed_before_nli(self):
        self.assertEqual(strip_citations("Claim [Doc 1, Doc 2]."),
                         "Claim .")

    def test_verifier_accepts_supported_cited_draft(self):
        docs = [{"doc_id": 1, "text": "Evidence supports the claim."}]
        self.assertTrue(verify_draft_citations(
            DummyVerifier(contradiction=0.05),
            "Evidence supports the claim [Doc 1].", docs
        ))

    def test_verifier_rejects_strong_contradiction(self):
        docs = [{"doc_id": 1, "text": "Evidence contradicts the claim."}]
        self.assertFalse(verify_draft_citations(
            DummyVerifier(contradiction=0.30),
            "Evidence supports the claim [Doc 1].", docs
        ))

    def test_verifier_rejects_unknown_citation(self):
        docs = [{"doc_id": 1, "text": "Evidence."}]
        self.assertFalse(verify_draft_citations(
            DummyVerifier(), "Claim [Doc 2].", docs
        ))

    def test_alternating_edge_order_is_deterministic(self):
        self.assertEqual(alternating_edge_order([10, 20, 30, 40, 50]),
                         [10, 30, 50, 40, 20])


if __name__ == "__main__":
    unittest.main()
