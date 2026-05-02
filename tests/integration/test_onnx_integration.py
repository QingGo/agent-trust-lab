"""Integration tests for ONNX-based NLI and semantic embedding.

All tests auto-skip when ONNX models are not cached.
"""

import pytest

pytestmark = pytest.mark.slow


def _skip_if_no_onnx():
    from tests.integration.conftest import _skip_if_missing_onnx

    _skip_if_missing_onnx()


class TestONNXIntegration:
    def test_embedding_engine_available(self):
        from agent_trust_lab.hallukg.anchoring import EmbeddingEngine

        engine = EmbeddingEngine()
        is_avail = engine.is_available
        if not is_avail:
            pytest.skip("ONNX embedding model not cached")

    def test_semantic_embedding_encodes_text(self):
        _skip_if_no_onnx()
        from agent_trust_lab.hallukg.anchoring import EmbeddingEngine

        engine = EmbeddingEngine()
        assert engine.is_available
        vec = engine.encode("email_send tool handles mailing.")
        assert vec is not None
        assert len(vec) > 0

    def test_anchoring_reasoner_real_onxx(self):
        _skip_if_no_onnx()
        from agent_trust_lab.hallukg import AnchoringReasoner

        reasoner = AnchoringReasoner()
        triples = [
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
        ]
        knowledge = "email_send tool handles mailing with to, subject, body, cc."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert len(results) == 1
        assert "label" in results[0]
        assert "anchor_score" in results[0]
        assert "evidence" in results[0]

    def test_faithfulness_checker_onnx_nli(self):
        _skip_if_no_onnx()
        from agent_trust_lab.hallukg import FaithfulnessChecker

        checker = FaithfulnessChecker()
        score = checker.check(
            ["The email_send function accepts a 'cc' parameter."],
            ["email_send tool handles mailing with to, subject, body, cc."],
        )
        assert 0.0 <= score <= 1.0

    def test_faithfulness_checker_batch(self):
        _skip_if_no_onnx()
        from agent_trust_lab.hallukg import FaithfulnessChecker

        checker = FaithfulnessChecker()
        scores = checker.batch_check(
            [
                "email_send takes a cc parameter.",
                "The database can be queried.",
            ],
            [
                "email_send tool handles mailing with cc.",
                "query_database searches records.",
            ],
        )
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_faithfulness_checker_empty_fallback(self):
        from agent_trust_lab.hallukg import FaithfulnessChecker

        checker = FaithfulnessChecker()
        score = checker.check([], [])
        assert 0.0 <= score <= 1.0

    def test_nli_neutral_weight_config(self):
        from agent_trust_lab.hallukg import FaithfulnessChecker

        checker = FaithfulnessChecker(nli_neutral_weight=0.8)
        assert checker.nli_neutral_weight == 0.8
