from unittest.mock import MagicMock, patch

import pytest

from agent_trust_lab.hallukg import (
    AnchoringReasoner,
    CodeHalluChecker,
    FaithfulnessChecker,
    GSARClassifier,
    TripleExtractor,
)
from agent_trust_lab.models.report import CodeHalluReport, HalluStepReport
from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep


class TestTripleExtractor:
    @pytest.fixture(autouse=True)
    def _stub_mode(self):
        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            yield

    def test_default_construction(self):
        extractor = TripleExtractor()
        assert extractor.model == "gpt-4o-mini"

    def test_custom_model_construction(self):
        extractor = TripleExtractor(model="custom-model")
        assert extractor.model == "custom-model"

    def test_model_name_backward_compat(self):
        extractor = TripleExtractor(model_name="legacy-model")
        assert extractor.model == "legacy-model"

    def test_extract_returns_list(self):
        extractor = TripleExtractor()
        result = extractor.extract("Hello world")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_extract_triple_structure(self):
        extractor = TripleExtractor()
        result = extractor.extract("Some text about agents")
        triple = result[0]
        assert triple["subject"] == "agent"
        assert triple["predicate"] == "generated_output"
        assert isinstance(triple["object"], str)
        assert triple["confidence"] == 0.85

    def test_extract_truncates_long_text(self):
        extractor = TripleExtractor()
        long_text = "A" * 150
        result = extractor.extract(long_text)
        triple = result[0]
        assert len(triple["object"]) <= 80

    def test_extract_empty_text(self):
        extractor = TripleExtractor()
        result = extractor.extract("")
        triple = result[0]
        assert triple["object"] == ""

    def test_extract_handles_newlines(self):
        extractor = TripleExtractor()
        result = extractor.extract("line1\nline2\nline3")
        triple = result[0]
        assert "\n" not in triple["object"]

    def test_extract_with_real_llm(self):
        """Verify real LLM path is called when API key is available."""
        from agent_trust_lab.hallukg.extractor import TripleEntry, TripleList

        mock_response = TripleList(
            triples=[TripleEntry(subject="agent", predicate="ran", object="test", confidence=0.9)]
        )

        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch("agent_trust_lab.llm.create_openai_client"):
                with patch("instructor.from_openai") as mock_instructor:
                    mock_client = mock_instructor.return_value
                    mock_client.chat.completions.create.return_value = mock_response
                    extractor = TripleExtractor(model="test-model")
                    result = extractor.extract("The agent ran the test")
                    assert len(result) == 1
                    assert result[0]["subject"] == "agent"
                    assert result[0]["predicate"] == "ran"
                    assert result[0]["confidence"] == 0.9

    def test_extract_fallback_to_stub_on_error(self):
        """Verify fallback to stub when LLM call raises exception."""
        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch(
                "agent_trust_lab.llm.create_openai_client", side_effect=Exception("API error")
            ):
                extractor = TripleExtractor()
                result = extractor.extract("test")
                assert len(result) == 1
                assert result[0]["subject"] == "agent"
                assert result[0]["confidence"] == 0.85


class TestAnchoringReasoner:
    @pytest.fixture(autouse=True)
    def _disable_onnx_embedding(self):
        """Force token overlap fallback by making EmbeddingEngine unavailable."""
        with patch(
            "agent_trust_lab.hallukg.anchoring.EmbeddingEngine.is_available",
            new_callable=lambda: property(lambda self: False),
        ):
            yield

    def test_default_construction(self):
        reasoner = AnchoringReasoner()
        assert reasoner.knowledge_base_path == "./kb/"
        assert reasoner.code_index_path is None

    def test_custom_construction(self):
        reasoner = AnchoringReasoner(
            knowledge_base_path="/custom/kb/",
            code_index_path="/custom/code/index",
        )
        assert reasoner.knowledge_base_path == "/custom/kb/"
        assert reasoner.code_index_path == "/custom/code/index"

    def test_anchor_returns_dict(self):
        reasoner = AnchoringReasoner()
        result = reasoner.anchor({"subject": "agent", "predicate": "test"})
        assert isinstance(result, dict)

    def test_anchor_returns_expected_keys(self):
        reasoner = AnchoringReasoner()
        result = reasoner.anchor({})
        assert result["label"] == "Grounded"
        assert "evidence" in result
        assert isinstance(result["evidence"], list)
        assert len(result["evidence"]) == 1
        assert result["anchor_score"] == 0.92

    def test_anchor_preserves_input_triple_fields(self):
        reasoner = AnchoringReasoner()
        triple = {
            "subject": "email_send",
            "predicate": "accepts",
            "object": "cc parameter",
            "confidence": 0.85,
        }
        result = reasoner.anchor(triple)
        assert result["subject"] == "email_send"
        assert result["predicate"] == "accepts"
        assert result["object"] == "cc parameter"
        assert result["confidence"] == 0.85

    def test_anchor_adds_grounding_fields(self):
        reasoner = AnchoringReasoner()
        triple = {"subject": "agent", "predicate": "ran", "object": "test"}
        result = reasoner.anchor(triple)
        assert result["label"] in ("Grounded", "Ungrounded")
        assert isinstance(result["evidence"], list)
        assert isinstance(result["anchor_score"], float)
        assert "subject" in result
        assert "predicate" in result
        assert "object" in result
        assert "confidence" in result

    def test_batch_anchor_returns_list(self):
        reasoner = AnchoringReasoner()
        triples = [{"s": "a"}, {"s": "b"}, {"s": "c"}]
        results = reasoner.batch_anchor(triples)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_anchor_all_grounded(self):
        reasoner = AnchoringReasoner()
        triples = [{"s": "a"}, {"s": "b"}]
        results = reasoner.batch_anchor(triples)
        for r in results:
            assert r["label"] == "Grounded"
            assert r["anchor_score"] == 0.92

    def test_batch_anchor_empty_list(self):
        reasoner = AnchoringReasoner()
        results = reasoner.batch_anchor([])
        assert results == []

    def test_batch_anchor_with_matching_knowledge_text(self):
        reasoner = AnchoringReasoner()
        triples = [
            {"subject": "email_send", "predicate": "accepts", "object": "cc", "confidence": 0.9},
            {"subject": "xyz_unknown", "predicate": "does", "object": "nothing", "confidence": 0.8},
        ]
        knowledge = "email_send accepts: to, subject, body, cc. dashboard_query: query monitoring."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert len(results) == 2
        assert results[0]["label"] == "Grounded"
        assert results[1]["label"] == "Ungrounded"

    def test_batch_anchor_ungrounded_when_no_match(self):
        reasoner = AnchoringReasoner()
        triples = [
            {
                "subject": "unknown_tool",
                "predicate": "does",
                "object": "nothing",
                "confidence": 0.5,
            },
        ]
        knowledge = "email_send accepts: to, subject, body."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert results[0]["label"] == "Ungrounded"
        assert results[0]["anchor_score"] == 0.15

    def test_batch_anchor_empty_knowledge_fallback(self):
        reasoner = AnchoringReasoner()
        triples = [
            {"subject": "anything", "predicate": "does", "object": "stuff", "confidence": 0.5},
        ]
        results = reasoner.batch_anchor(triples, knowledge_text="")
        assert results[0]["label"] == "Grounded"
        assert results[0]["anchor_score"] == 0.92

    def test_batch_anchor_object_match(self):
        reasoner = AnchoringReasoner()
        triples = [
            {
                "subject": "unknown_sig",
                "predicate": "has",
                "object": "email_send",
                "confidence": 0.7,
            },
        ]
        knowledge = "email_send accepts: to, subject, body."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert results[0]["label"] == "Grounded"

    def test_batch_anchor_case_insensitive_match(self):
        reasoner = AnchoringReasoner()
        triples = [
            {"subject": "Email_Send", "predicate": "accepts", "object": "CC", "confidence": 0.9},
        ]
        knowledge = "email_send accepts: to, subject, body."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert results[0]["label"] == "Grounded"

    def test_batch_anchor_preserves_fields_with_knowledge(self):
        reasoner = AnchoringReasoner()
        triples = [
            {"subject": "foo", "predicate": "bar", "object": "baz", "confidence": 0.99},
        ]
        results = reasoner.batch_anchor(triples, knowledge_text="foo bar baz")
        assert results[0]["subject"] == "foo"
        assert results[0]["predicate"] == "bar"
        assert results[0]["object"] == "baz"
        assert results[0]["confidence"] == 0.99
        assert results[0]["label"] == "Grounded"

    def test_custom_grounded_threshold_setting(self):
        reasoner = AnchoringReasoner(grounded_threshold=0.7)
        assert reasoner.grounded_threshold == 0.7

    def test_custom_grounded_threshold_high_value_ungrounds(self):
        reasoner = AnchoringReasoner(grounded_threshold=0.95)
        triples = [
            {"subject": "email_send", "predicate": "accepts", "object": "cc", "confidence": 0.9},
        ]
        knowledge = "email_send accepts: to, subject, body, cc."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert results[0]["label"] == "Ungrounded"

    def test_custom_grounded_threshold_low_value_grounds(self):
        reasoner = AnchoringReasoner(grounded_threshold=0.1)
        triples = [
            {
                "subject": "unknown_tool",
                "predicate": "does",
                "object": "nothing",
                "confidence": 0.5,
            },
        ]
        knowledge = "email_send accepts: to, subject, body."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert results[0]["label"] == "Grounded"


class TestAnchoringReasonerSemantic:
    """Tests for the ONNX semantic embedding path."""

    @pytest.fixture(autouse=True)
    def _mock_embedding_engine(self):
        """Mock EmbeddingEngine so we control embeddings deterministically."""
        with patch("agent_trust_lab.hallukg.anchoring.EmbeddingEngine") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.is_available = True
            yield mock_instance

    def _setup_embeddings(self, mock_instance, embeddings: dict):
        """Configure mock encode() to return specific embedding vectors."""

        def encode_side_effect(text):
            import numpy as np

            for key, vec in embeddings.items():
                if key in text:
                    return np.array(vec, dtype=np.float64)
            return np.array([0.0] * 4, dtype=np.float64)

        mock_instance.encode.side_effect = encode_side_effect

    def test_semantic_anchor_matching(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triple = {"subject": "email_send", "predicate": "accepts", "object": "cc"}
        knowledge = "email_send tool handles mailing. database_query tool searches."
        result = reasoner.anchor(triple, knowledge_text=knowledge)
        assert result["label"] == "Grounded"
        assert result["anchor_score"] > 0.5
        assert "Semantic match" in result["evidence"][0]

    def test_semantic_anchor_no_match(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
            "unknown_tool": [0.0, 1.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triple = {"subject": "unknown_tool", "predicate": "does", "object": "nothing"}
        knowledge = "email_send tool handles mailing."
        result = reasoner.anchor(triple, knowledge_text=knowledge)
        assert result["label"] == "Ungrounded"
        assert result["anchor_score"] < 0.3

    def test_semantic_score_is_continuous(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triple = {"subject": "email_send", "predicate": "accepts", "object": "cc"}
        knowledge = "email_send tool handles mailing."
        result = reasoner.anchor(triple, knowledge_text=knowledge)
        assert isinstance(result["anchor_score"], float)
        assert 0.0 <= result["anchor_score"] <= 1.0

    def test_semantic_batch_anchor(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
            "unknown_tool": [0.0, 1.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triples = [
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
            {"subject": "unknown_tool", "predicate": "does", "object": "nothing"},
        ]
        knowledge = "email_send tool handles mailing. database_query tool searches."
        results = reasoner.batch_anchor(triples, knowledge_text=knowledge)
        assert len(results) == 2
        assert results[0]["label"] == "Grounded"
        assert results[1]["label"] == "Ungrounded"

    def test_semantic_preserves_triple_fields(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triple = {
            "subject": "email_send",
            "predicate": "accepts",
            "object": "cc",
            "confidence": 0.99,
        }
        knowledge = "email_send tool handles mailing."
        result = reasoner.anchor(triple, knowledge_text=knowledge)
        assert result["subject"] == "email_send"
        assert result["predicate"] == "accepts"
        assert result["object"] == "cc"
        assert result["confidence"] == 0.99

    def test_semantic_evidence_references_best_match(self, _mock_embedding_engine):
        reasoner = AnchoringReasoner()
        embeddings = {
            "email_send": [1.0, 0.0, 0.0, 0.0],
        }
        self._setup_embeddings(_mock_embedding_engine, embeddings)

        triple = {"subject": "email_send", "predicate": "accepts", "object": "cc"}
        knowledge = "email_send tool handles mailing with to, subject, body, cc."
        result = reasoner.anchor(triple, knowledge_text=knowledge)
        assert len(result["evidence"]) > 0
        assert "email_send" in result["evidence"][0]


class TestGSARClassifier:
    @pytest.fixture(autouse=True)
    def _stub_mode(self):
        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            yield

    def test_classify_returns_list_of_hallu_step_reports(self):
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="step")]
        reports = classifier.classify(steps, [])
        assert isinstance(reports, list)
        assert len(reports) == 1
        assert isinstance(reports[0], HalluStepReport)

    def test_classify_cycles_gsar_labels(self):
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content=f"step{i}") for i in range(5)]
        reports = classifier.classify(steps, [])
        labels = [r.gsar_label for r in reports]
        assert labels == ["Grounded", "Grounded", "Complementary", "Ungrounded", "Contradicted"]

    def test_classify_step_indices(self):
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="s") for _ in range(3)]
        reports = classifier.classify(steps, [])
        for i, report in enumerate(reports):
            assert report.step_index == i

    def test_classify_grounded_scores(self):
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="s")]
        reports = classifier.classify(steps, [])
        assert reports[0].g_score == 0.7
        assert reports[0].u_score == 0.1
        assert reports[0].c_score == 0.2
        assert reports[0].faithfulness_score == 0.95

    def test_classify_complementary_scores(self):
        classifier = GSARClassifier()
        steps = [
            TrajectoryStep(type="thought", content="s"),
            TrajectoryStep(type="thought", content="s"),
            TrajectoryStep(type="thought", content="s"),
        ]
        reports = classifier.classify(steps, [])
        complementary = reports[2]
        assert complementary.gsar_label == "Complementary"
        assert complementary.g_score == 0.3

    def test_classify_empty_steps(self):
        classifier = GSARClassifier()
        reports = classifier.classify([], [])
        assert reports == []

    def test_classify_has_evidence_and_explanation(self):
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="step")]
        reports = classifier.classify(steps, [])
        assert len(reports[0].evidence) == 1
        assert "stub" in reports[0].explanation.lower()

    def test_model_parameter(self):
        classifier = GSARClassifier(model="custom-gsar-model")
        assert classifier.model == "custom-gsar-model"

    def test_default_model(self):
        classifier = GSARClassifier()
        assert classifier.model == "deepseek-v4-flash"

    def test_classify_with_real_llm(self):
        """Verify real LLM path is called when API key is available."""
        from agent_trust_lab.hallukg.classifier import GSAROutput, GSARStepResult

        mock_response = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0,
                    gsar_label="Grounded",
                    g_score=0.9,
                    u_score=0.1,
                    c_score=0.0,
                    faithfulness_score=0.95,
                    evidence=["anchored triple match"],
                    explanation="Fully grounded in triples",
                )
            ]
        )

        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch("agent_trust_lab.llm.create_openai_client"):
                with patch("instructor.from_openai") as mock_instructor:
                    mock_client = mock_instructor.return_value
                    mock_client.chat.completions.create.return_value = mock_response
                    classifier = GSARClassifier(model="test-model")
                    steps = [TrajectoryStep(type="thought", content="test")]
                    result = classifier.classify(steps, [])
                    assert len(result) == 1
                    assert result[0].step_index == 0
                    assert result[0].gsar_label == "Grounded"
                    assert result[0].g_score == 0.9

    def test_classify_fallback_to_stub_on_error(self):
        """Verify fallback to stub when LLM call raises exception."""
        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch(
                "agent_trust_lab.llm.create_openai_client", side_effect=Exception("API down")
            ):
                classifier = GSARClassifier()
                steps = [TrajectoryStep(type="thought", content="s")]
                result = classifier.classify(steps, [])
                assert len(result) == 1
                assert "stub" in result[0].explanation.lower()

    def test_classify_multi_model_single_model(self):
        """Single model in model_list should behave like classify()."""
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="step")]
        result = classifier.classify_multi_model(steps, [], ["deepseek-v4-flash"])
        assert len(result) == 1
        assert result[0].step_index == 0

    def test_classify_multi_model_empty_list_falls_back(self):
        """Empty model_list should delegate to classify()."""
        classifier = GSARClassifier()
        steps = [TrajectoryStep(type="thought", content="step")]
        result = classifier.classify_multi_model(steps, [], [])
        assert len(result) == 1

    def test_classify_multi_model_voting(self):
        """Multi-model voting should merge results from multiple models."""
        from agent_trust_lab.hallukg.classifier import GSAROutput, GSARStepResult

        model_a_response = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0,
                    gsar_label="Grounded",
                    g_score=0.9,
                    u_score=0.05,
                    c_score=0.0,
                    faithfulness_score=0.95,
                    evidence=["a evidence"],
                    explanation="model A grounded",
                ),
            ]
        )
        model_b_response = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0,
                    gsar_label="Grounded",
                    g_score=0.7,
                    u_score=0.2,
                    c_score=0.1,
                    faithfulness_score=0.8,
                    evidence=["b evidence"],
                    explanation="model B grounded",
                ),
            ]
        )
        model_c_response = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0,
                    gsar_label="Ungrounded",
                    g_score=0.1,
                    u_score=0.9,
                    c_score=0.0,
                    faithfulness_score=0.3,
                    evidence=["c evidence"],
                    explanation="model C ungrounded",
                ),
            ]
        )

        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch("agent_trust_lab.llm.create_openai_client"):
                with patch("instructor.from_openai") as mock_instructor:
                    mock_client = mock_instructor.return_value
                    mock_client.chat.completions.create.side_effect = [
                        model_a_response,
                        model_b_response,
                        model_c_response,
                    ]
                    classifier = GSARClassifier()
                    steps = [TrajectoryStep(type="thought", content="step")]
                    result = classifier.classify_multi_model(
                        steps, [], ["model-a", "model-b", "model-c"]
                    )
                    assert len(result) == 1
                    assert result[0].gsar_label == "Grounded"
                    assert 0.5 < result[0].g_score < 0.9
                    assert "Multi-model vote" in result[0].explanation

    def test_classify_multi_model_majority_ungrounded(self):
        """Majority vote should select the label with most votes."""
        from agent_trust_lab.hallukg.classifier import GSAROutput, GSARStepResult

        u1 = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0, gsar_label="Ungrounded", g_score=0.1, u_score=0.9,
                    c_score=0.0, faithfulness_score=0.2, explanation="u1",
                ),
            ]
        )
        u2 = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0, gsar_label="Ungrounded", g_score=0.0, u_score=1.0,
                    c_score=0.0, faithfulness_score=0.1, explanation="u2",
                ),
            ]
        )
        g1 = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0, gsar_label="Grounded", g_score=0.9, u_score=0.0,
                    c_score=0.0, faithfulness_score=0.9, explanation="g1",
                ),
            ]
        )

        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch("agent_trust_lab.llm.create_openai_client"):
                with patch("instructor.from_openai") as mock_instructor:
                    mock_client = mock_instructor.return_value
                    mock_client.chat.completions.create.side_effect = [u1, u2, g1]
                    classifier = GSARClassifier()
                    steps = [TrajectoryStep(type="thought", content="step")]
                    result = classifier.classify_multi_model(
                        steps, [], ["m1", "m2", "m3"]
                    )
                    assert result[0].gsar_label == "Ungrounded"

    def test_classify_multi_model_all_fail_fallback(self):
        """When all real LLM calls fail, each model uses stub internally and voting still works."""
        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch(
                "agent_trust_lab.llm.create_openai_client",
                side_effect=Exception("All down"),
            ):
                classifier = GSARClassifier()
                steps = [TrajectoryStep(type="thought", content="step")]
                result = classifier.classify_multi_model(
                    steps, [], ["m1", "m2"]
                )
                assert len(result) == 1
                assert "Multi-model vote" in result[0].explanation
                assert result[0].gsar_label is not None

    def test_classify_multi_model_preserves_step_indices(self):
        """Multi-model voting should preserve correct step indices for multiple steps."""
        from agent_trust_lab.hallukg.classifier import GSAROutput, GSARStepResult

        response = GSAROutput(
            steps=[
                GSARStepResult(
                    step_index=0, gsar_label="Grounded", g_score=0.8, u_score=0.1,
                    c_score=0.05, faithfulness_score=0.9, explanation="step0",
                ),
                GSARStepResult(
                    step_index=1, gsar_label="Ungrounded", g_score=0.2, u_score=0.7,
                    c_score=0.3, faithfulness_score=0.4, explanation="step1",
                ),
                GSARStepResult(
                    step_index=2, gsar_label="Complementary", g_score=0.5, u_score=0.3,
                    c_score=0.2, faithfulness_score=0.6, explanation="step2",
                ),
            ]
        )

        with patch("agent_trust_lab.llm.get_api_key", return_value="mock-key"):
            with patch("agent_trust_lab.llm.create_openai_client"):
                with patch("instructor.from_openai") as mock_instructor:
                    mock_client = mock_instructor.return_value
                    mock_client.chat.completions.create.side_effect = [response, response]
                    classifier = GSARClassifier()
                    steps = [
                        TrajectoryStep(type="thought", content="s0"),
                        TrajectoryStep(type="thought", content="s1"),
                        TrajectoryStep(type="thought", content="s2"),
                    ]
                    result = classifier.classify_multi_model(steps, [], ["m1", "m2"])
                    assert len(result) == 3
                    assert result[0].step_index == 0
                    assert result[1].step_index == 1
                    assert result[2].step_index == 2


class TestFaithfulnessChecker:
    @pytest.fixture(autouse=True)
    def _disable_onnx_nli(self):
        with patch.object(FaithfulnessChecker, "_check_onnx", return_value=False):
            yield

    def test_check_returns_float_between_0_and_1(self):
        checker = FaithfulnessChecker()
        score = checker.check(["the agent called the database_query tool"], ["database_query"])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_check_identical_texts(self):
        checker = FaithfulnessChecker()
        text = "the agent executed the file_read tool with the correct parameters"
        score = checker.check([text], [text])
        assert score == 1.0

    def test_check_different_texts(self):
        checker = FaithfulnessChecker()
        score = checker.check(
            ["database_query accepts limit and format parameters"],
            ["file_write accepts path and content parameters"],
        )
        assert score < 1.0

    def test_check_empty_statements(self):
        checker = FaithfulnessChecker()
        score = checker.check([""], ["evidence"])
        assert score == 0.5

    def test_check_empty_evidence(self):
        checker = FaithfulnessChecker()
        score = checker.check(["statement"], [""])
        assert score == 0.5

    def test_check_both_empty(self):
        checker = FaithfulnessChecker()
        score = checker.check([], [])
        assert score == 0.5

    def test_check_similar_texts(self):
        checker = FaithfulnessChecker()
        score = checker.check(
            ["the agent used database_query with limit parameter"],
            ["database_query tool accepts limit parameter for queries"],
        )
        assert score > 0.0

    def test_batch_check_returns_correct_length(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check(
            [["s1"], ["s2"], ["s3"]],
            [["e1"], ["e2"], ["e3"]],
        )
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_batch_check_empty(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check([], [])
        assert scores == []

    def test_batch_check_mixed_similarity(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check(
            [["identical"], ["different"], ["also different"]],
            [["identical"], ["other"], ["something else"]],
        )
        assert len(scores) == 3
        assert scores[0] == 1.0
        assert scores[1] < 1.0
        assert scores[2] < 1.0

    def test_batch_check_with_empty_entries(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check(
            [["text"], [""], ["more text"]],
            [["evidence"], ["ev"], ["evidence"]],
        )
        assert len(scores) == 3
        assert scores[1] == 0.5

    def test_check_with_realistic_evidence(self):
        checker = FaithfulnessChecker()
        score = checker.check(
            ["The email_send function accepts to, subject, body, and cc parameters."],
            ["email_send accepts: to, subject, body, cc. cc is an optional field."],
        )
        assert score > 0.2

    def test_custom_nli_neutral_weight_setting(self):
        checker = FaithfulnessChecker(nli_neutral_weight=0.8)
        assert checker.nli_neutral_weight == 0.8

    def test_default_nli_neutral_weight(self):
        checker = FaithfulnessChecker()
        assert checker.nli_neutral_weight == 0.5


class TestCodeHalluChecker:
    @pytest.fixture(autouse=True)
    def _stub_mode(self):
        with patch(
            "agent_trust_lab.sandbox.image.get_docker_client",
            side_effect=Exception("No Docker daemon"),
        ):
            yield

    def test_default_construction(self):
        checker = CodeHalluChecker()
        assert checker.timeout == 30
        assert checker.docker_host == ""
        assert checker.python_image == "docker.m.daocloud.io/library/python:3-slim"

    def test_custom_timeout_construction(self):
        checker = CodeHalluChecker(timeout=60)
        assert checker.timeout == 60

    def test_custom_docker_host(self):
        checker = CodeHalluChecker(docker_host="unix:///custom.sock")
        assert checker.docker_host == "unix:///custom.sock"

    def test_custom_python_image(self):
        checker = CodeHalluChecker(python_image="python:3.12-alpine")
        assert checker.python_image == "python:3.12-alpine"

    def test_check_returns_code_hallu_report(self):
        checker = CodeHalluChecker()
        report = checker.check(code="import os", test_command="python test.py")
        assert isinstance(report, CodeHalluReport)

    def test_check_syntax_error_detection(self):
        checker = CodeHalluChecker()
        report = checker.check("x =")
        assert report.hallucination_type == "logic_hallucination"
        assert "SyntaxError" in (report.error_message or "")

    def test_check_stub_fallback(self):
        checker = CodeHalluChecker()
        report = checker.check("import os")
        assert report.hallucination_type == "naming"
        assert "ImportError" in (report.error_message or "")
        assert "stub" in (report.error_message or "").lower()

    def test_check_truncates_code_snippet(self):
        checker = CodeHalluChecker()
        long_code = "import os\n" * 50
        report = checker.check(long_code)
        assert len(report.code_snippet) <= 100

    def test_check_preserves_expected_error(self):
        checker = CodeHalluChecker()
        report = checker.check("import os", expected_error="ImportError")
        assert report.expected_error_pattern == "ImportError"

    def test_batch_check_filters_steps(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="thought", content="thinking"),
            TrajectoryStep(type="code_generation", content="import os"),
            TrajectoryStep(type="observation", content="result"),
            TrajectoryStep(type="trap_injection", content="trap1"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        assert len(reports) == 1
        assert reports[0].step_index == 1

    def test_batch_check_multiple_code_steps(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="code_generation", content="import os"),
            TrajectoryStep(type="code_generation", content="import sys"),
            TrajectoryStep(type="code_generation", content="import json"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        assert len(reports) == 3
        assert reports[0].step_index == 0
        assert reports[1].step_index == 1
        assert reports[2].step_index == 2

    def test_batch_check_empty_trajectory(self):
        checker = CodeHalluChecker()
        trajectory = SecureTrajectory(steps=[], security_events=[])
        reports = checker.batch_check(trajectory)
        assert reports == []

    def test_batch_check_no_code_steps(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="thought", content="t1"),
            TrajectoryStep(type="observation", content="o1"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        assert reports == []

    def test_check_has_fix_suggestion(self):
        checker = CodeHalluChecker()
        report = checker.check("import os")
        assert len(report.fix_suggestion or "") > 0

    def test_check_has_error_message(self):
        checker = CodeHalluChecker()
        report = checker.check("import os")
        assert "Error" in (report.error_message or "")

    def test_classify_error_mapping(self):
        checker = CodeHalluChecker()
        assert checker._classify_error("ImportError", "") == "mapping"
        assert checker._classify_error("ModuleNotFoundError", "") == "mapping"
        assert checker._classify_error("FileNotFoundError", "") == "mapping"

    def test_classify_error_naming(self):
        checker = CodeHalluChecker()
        assert checker._classify_error("AttributeError", "") == "naming"
        assert checker._classify_error("NameError", "") == "naming"

    def test_classify_error_parameter(self):
        checker = CodeHalluChecker()
        assert checker._classify_error("TypeError", "") == "parameter"
        assert checker._classify_error("ValueError", "") == "parameter"

    def test_classify_error_logic(self):
        checker = CodeHalluChecker()
        assert checker._classify_error("SyntaxError", "") == "logic_hallucination"
        assert checker._classify_error("IndentationError", "") == "logic_hallucination"
        assert checker._classify_error("UnknownError", "") == "logic_hallucination"

    def test_parse_error_extracts_name_and_message(self):
        checker = CodeHalluChecker()
        name, msg = checker._parse_error("Traceback:\n  File ...\nImportError: No module named foo")
        assert name == "ImportError"
        assert "foo" in (msg or "")

    def test_parse_error_no_match(self):
        checker = CodeHalluChecker()
        name, msg = checker._parse_error("some unexpected output")
        assert name is None

    def test_suggest_fix_mapping(self):
        checker = CodeHalluChecker()
        fix = checker._suggest_fix("mapping", "ImportError", "")
        assert "library" in fix.lower()

    def test_suggest_fix_naming(self):
        checker = CodeHalluChecker()
        fix = checker._suggest_fix("naming", "AttributeError", "")
        assert "api" in fix.lower()

    def test_suggest_fix_parameter(self):
        checker = CodeHalluChecker()
        fix = checker._suggest_fix("parameter", "TypeError", "")
        assert "parameter" in fix.lower()

    def test_successful_execution_no_error(self):
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"hello world\n"
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=mock_client):
            with patch("agent_trust_lab.sandbox.image.ImageManager") as mock_img_mgr:
                mock_img_mgr.return_value.ensure_image.return_value = True
                checker = CodeHalluChecker()
                report = checker.check("print('hello')")
                assert report.hallucination_type == "none"
                assert report.error_message is None
                assert report.fix_suggestion is None

    def test_docker_execution_with_import_error(self):
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = (
            b"Traceback (most recent call last):\n"
            b"  File '<string>', line 1, in <module>\n"
            b"ModuleNotFoundError: No module named 'fake_lib'\n"
        )
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=mock_client):
            with patch("agent_trust_lab.sandbox.image.ImageManager") as mock_img_mgr:
                mock_img_mgr.return_value.ensure_image.return_value = True
                checker = CodeHalluChecker()
                report = checker.check("import fake_lib")
                assert report.hallucination_type == "mapping"
                assert "ModuleNotFoundError" in (report.error_message or "")
                assert "fake_lib" in (report.error_message or "")

    def test_docker_execution_with_attribute_error(self):
        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = (
            b"Traceback (most recent call last):\n"
            b"  File '<string>', line 1\n"
            b"AttributeError: 'list' object has no attribute 'notexist'\n"
        )
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=mock_client):
            with patch("agent_trust_lab.sandbox.image.ImageManager") as mock_img_mgr:
                mock_img_mgr.return_value.ensure_image.return_value = True
                checker = CodeHalluChecker()
                report = checker.check("[].notexist()")
                assert report.hallucination_type == "naming"
                assert "AttributeError" in (report.error_message or "")


class TestKnowledgeGraph:
    def test_default_construction(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        assert kg.edges == 0
        assert kg.size() == 0

    def test_add_single_triple(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b"})
        assert kg.entity_exists("a")
        assert kg.entity_exists("b")
        assert kg.edges == 1

    def test_add_multiple_triples(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triples(
            [
                {"subject": "a", "predicate": "p", "object": "b"},
                {"subject": "b", "predicate": "q", "object": "c"},
            ]
        )
        assert kg.size() >= 3
        assert kg.edges == 2

    def test_add_triple_empty_fields_skipped(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "", "predicate": "p", "object": "b"})
        assert kg.edges == 0
        kg.add_triple({"subject": "a", "predicate": "p", "object": ""})
        assert kg.edges == 0

    def test_find_direct_path(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b"})
        path = kg.find_shortest_path("a", "b")
        assert path is not None
        assert len(path) == 2

    def test_find_multi_hop_path(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triples(
            [
                {"subject": "a", "predicate": "p", "object": "b"},
                {"subject": "b", "predicate": "q", "object": "c"},
                {"subject": "c", "predicate": "r", "object": "d"},
            ]
        )
        path = kg.find_shortest_path("a", "d")
        assert path is not None
        assert len(path) == 4

    def test_no_path_returns_none(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b"})
        path = kg.find_shortest_path("a", "x")
        assert path is None

    def test_path_exceeds_max_hops(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triples(
            [
                {"subject": "a", "predicate": "p", "object": "b"},
                {"subject": "b", "predicate": "q", "object": "c"},
                {"subject": "c", "predicate": "r", "object": "d"},
            ]
        )
        path = kg.find_shortest_path("a", "d", max_hops=1)
        assert path is None

    def test_self_path(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b"})
        path = kg.find_shortest_path("a", "a")
        assert path is not None
        assert len(path) == 1

    def test_get_edge_data(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b", "confidence": 0.9})
        edge = kg.get_edge_data("a", "b")
        assert edge is not None
        assert edge["predicate"] == "p"
        assert edge["confidence"] == 0.9

    def test_get_edge_data_missing(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        edge = kg.get_edge_data("x", "y")
        assert edge is None

    def test_neighbors(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "a", "predicate": "p", "object": "b"})
        kg.add_triple({"subject": "a", "predicate": "q", "object": "c"})
        neighbors = kg.neighbors("a")
        assert len(neighbors) == 2
        assert "b" in neighbors
        assert "c" in neighbors

    def test_neighbors_unknown_entity(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        neighbors = kg.neighbors("nonexistent")
        assert neighbors == []

    def test_entity_exists_case_insensitive(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "EmailSend", "predicate": "p", "object": "MailHandler"})
        assert kg.entity_exists("emailsend")
        assert kg.entity_exists("EmAiLsEnD")

    def test_add_knowledge_text(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_text("email_send tool handles mailing with cc.")
        assert kg.size() > 0
        assert kg.edges > 0

    def test_add_knowledge_text_empty(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_text("")
        assert kg.size() == 0

    def test_stop_word_filtering(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_text(
            "the email_send tool handles mailing with the cc parameter"
        )
        assert kg.size() > 0
        assert kg.entity_exists("email_send") or kg.entity_exists("email")

    def test_stop_words_removed_from_edges(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_sentence("the tool is for testing")
        assert not kg.entity_exists("the")
        assert not kg.entity_exists("is")

    def test_sentence_with_only_stop_words(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_sentence("the and for with")
        assert kg.size() == 0

    def test_entity_resolve_exact_match(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "email_send", "predicate": "accepts", "object": "cc"})
        resolved = kg.entity_resolve("email_send")
        assert resolved is not None

    def test_entity_resolve_normalized_match(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "emails", "predicate": "use", "object": "servers"})
        resolved = kg.entity_resolve("emails")
        assert resolved is not None

    def test_entity_resolve_no_match(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "x", "predicate": "p", "object": "y"})
        resolved = kg.entity_resolve("no_such_entity_xyz")
        assert resolved is None

    def test_normalize_entity_plural(self):
        from agent_trust_lab.hallukg.multi_hop import _normalize_entity

        assert _normalize_entity("tools") == "tool"

    def test_normalize_entity_suffix(self):
        from agent_trust_lab.hallukg.multi_hop import _normalize_entity

        assert _normalize_entity("testing") == "test"
        assert _normalize_entity("handled") == "handl"

    def test_normalize_entity_short_word(self):
        from agent_trust_lab.hallukg.multi_hop import _normalize_entity

        assert _normalize_entity("is") == "is"
        assert _normalize_entity("a") == "a"

    def test_add_knowledge_text_stop_word_edges_filtered(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_knowledge_text(
            "The email_send tool is for sending emails. The database_query tool is for data."
        )
        assert kg.size() > 0
        assert not kg.entity_exists("the")
        assert not kg.entity_exists("is")
        assert not kg.entity_exists("for")

    def test_entity_resolve_substring_match(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "email_send", "predicate": "handles", "object": "mailing"})
        resolved = kg.entity_resolve("email")
        assert resolved == "email" or resolved is not None


class TestMultiHopReasoner:
    def test_default_construction(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        assert mh.grounded_threshold == 0.3
        assert mh.max_hops == 3

    def test_custom_construction(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner(grounded_threshold=0.5, max_hops=2)
        assert mh.grounded_threshold == 0.5
        assert mh.max_hops == 2

    def test_anchor_direct_hit(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph, MultiHopReasoner

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "email_send", "predicate": "accepts", "object": "cc"})
        mh = MultiHopReasoner(knowledge_graph=kg)

        result = mh.anchor(
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
        )
        assert result["label"] == "Grounded"
        assert result["anchor_score"] == 0.95
        assert result["multi_hop"] is True
        assert result["hop_count"] == 1

    def test_anchor_multi_hop_path(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph, MultiHopReasoner

        kg = KnowledgeGraph()
        kg.add_triples(
            [
                {"subject": "email_send", "predicate": "calls", "object": "smtp_client"},
                {"subject": "smtp_client", "predicate": "uses", "object": "cc_header"},
            ]
        )
        mh = MultiHopReasoner(knowledge_graph=kg)

        result = mh.anchor(
            {"subject": "email_send", "predicate": "affects", "object": "cc_header"},
        )
        assert result["multi_hop"] is True
        assert result["hop_count"] >= 1

    def test_anchor_no_path(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph, MultiHopReasoner

        kg = KnowledgeGraph()
        kg.add_triple({"subject": "x", "predicate": "p", "object": "y"})
        mh = MultiHopReasoner(knowledge_graph=kg)

        result = mh.anchor(
            {"subject": "a", "predicate": "p", "object": "b"},
        )
        assert result["label"] == "Ungrounded"
        assert result["anchor_score"] == 0.0
        assert result["hop_count"] == 0

    def test_anchor_partial_knowledge_match(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        result = mh.anchor(
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
            knowledge_text="email_send tool handles mailing.",
        )
        assert result["multi_hop"] is True
        assert "email_send" in str(result["evidence"])

    def test_anchor_both_match_knowledge(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        result = mh.anchor(
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
            knowledge_text="email_send tool handles mailing with cc.",
        )
        assert result["anchor_score"] >= 0.4

    def test_batch_anchor_returns_list(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        triples = [
            {"subject": "email_send", "predicate": "accepts", "object": "cc"},
            {"subject": "db", "predicate": "supports", "object": "sql"},
        ]
        results = mh.batch_anchor(
            triples,
            knowledge_text="email_send tool handles mailing with cc.",
        )
        assert len(results) == 2
        for r in results:
            assert "label" in r
            assert "anchor_score" in r
            assert "evidence" in r

    def test_batch_anchor_empty(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        results = mh.batch_anchor([])
        assert results == []

    def test_batch_anchor_builds_graph(self):
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        mh = MultiHopReasoner()
        triples = [
            {"subject": "a", "predicate": "p", "object": "b"},
            {"subject": "b", "predicate": "q", "object": "c"},
        ]
        results = mh.batch_anchor(triples)
        assert len(results) == 2
        assert mh.knowledge_graph.edges >= 2

    def test_anchor_with_custom_knowledge_graph(self):
        from agent_trust_lab.hallukg.multi_hop import KnowledgeGraph, MultiHopReasoner

        kg = KnowledgeGraph()
        kg.add_triples(
            [
                {"subject": "func_a", "predicate": "returns", "object": "result_x"},
                {"subject": "result_x", "predicate": "used_by", "object": "func_b"},
            ]
        )
        mh = MultiHopReasoner(knowledge_graph=kg, max_hops=3)
        result = mh.anchor(
            {"subject": "func_a", "predicate": "impacts", "object": "func_b"},
        )
        assert result["multi_hop"] is True
        assert result["hop_count"] >= 2
        assert result["anchor_score"] > 0.0

    def test_merge_anchor_results_prefers_higher_score(self):
        from agent_trust_lab.orchestrator import Orchestrator

        single = [
            {
                "subject": "a",
                "predicate": "p",
                "object": "b",
                "label": "Ungrounded",
                "anchor_score": 0.15,
                "evidence": ["s1"],
            },
        ]
        multi = [
            {
                "subject": "a",
                "predicate": "p",
                "object": "b",
                "label": "Grounded",
                "anchor_score": 0.95,
                "evidence": ["m1"],
                "multi_hop": True,
                "hop_count": 2,
            },
        ]
        merged = Orchestrator._merge_anchor_results(single, multi)
        assert len(merged) == 1
        assert merged[0]["anchor_score"] == 0.95
        assert merged[0]["label"] == "Grounded"

    def test_merge_anchor_results_keeps_single_when_higher(self):
        from agent_trust_lab.orchestrator import Orchestrator

        single = [
            {
                "subject": "a",
                "predicate": "p",
                "object": "b",
                "label": "Grounded",
                "anchor_score": 0.92,
                "evidence": ["s1"],
            },
        ]
        multi = [
            {
                "subject": "a",
                "predicate": "p",
                "object": "b",
                "label": "Ungrounded",
                "anchor_score": 0.1,
                "evidence": ["m1"],
                "multi_hop": True,
                "hop_count": 0,
            },
        ]
        merged = Orchestrator._merge_anchor_results(single, multi)
        assert len(merged) == 1
        assert merged[0]["anchor_score"] == 0.92

    def test_merge_anchor_results_length_mismatch(self):
        from agent_trust_lab.orchestrator import Orchestrator

        single = [{"anchor_score": 0.5}]
        multi = [{"anchor_score": 0.5}, {"anchor_score": 0.5}]
        merged = Orchestrator._merge_anchor_results(single, multi)
        assert merged is single
