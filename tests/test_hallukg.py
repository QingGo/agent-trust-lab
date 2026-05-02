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
            triples=[
                TripleEntry(subject="agent", predicate="ran", object="test", confidence=0.9)
            ]
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
            with patch("agent_trust_lab.llm.create_openai_client",
                       side_effect=Exception("API error")):
                extractor = TripleExtractor()
                result = extractor.extract("test")
                assert len(result) == 1
                assert result[0]["subject"] == "agent"
                assert result[0]["confidence"] == 0.85


class TestAnchoringReasoner:
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
            {"subject": "unknown_tool", "predicate": "does", "object": "nothing",
             "confidence": 0.5},
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
            {"subject": "unknown_sig", "predicate": "has", "object": "email_send",
             "confidence": 0.7},
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
        assert labels == [
            "Grounded", "Grounded", "Complementary", "Ungrounded", "Contradicted"
        ]

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
            with patch("agent_trust_lab.llm.create_openai_client",
                       side_effect=Exception("API down")):
                classifier = GSARClassifier()
                steps = [TrajectoryStep(type="thought", content="s")]
                result = classifier.classify(steps, [])
                assert len(result) == 1
                assert "stub" in result[0].explanation.lower()


class TestFaithfulnessChecker:
    def test_check_returns_constant(self):
        checker = FaithfulnessChecker()
        score = checker.check(["statement"], ["evidence"])
        assert score == 0.95

    def test_check_with_multiple_statements(self):
        checker = FaithfulnessChecker()
        score = checker.check(["s1", "s2", "s3"], ["e1", "e2"])
        assert score == 0.95

    def test_batch_check_returns_correct_length(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check(
            [["s1"], ["s2"], ["s3"]],
            [["e1"], ["e2"], ["e3"]],
        )
        assert len(scores) == 3
        assert scores == [0.95, 0.95, 0.95]

    def test_batch_check_empty(self):
        checker = FaithfulnessChecker()
        scores = checker.batch_check([], [])
        assert scores == []


class TestCodeHalluChecker:
    @pytest.fixture(autouse=True)
    def _stub_mode(self):
        with patch("agent_trust_lab.sandbox.image.get_docker_client",
                   side_effect=Exception("No Docker daemon")):
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

        with patch("agent_trust_lab.sandbox.image.get_docker_client",
                   return_value=mock_client):
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

        with patch("agent_trust_lab.sandbox.image.get_docker_client",
                   return_value=mock_client):
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

        with patch("agent_trust_lab.sandbox.image.get_docker_client",
                   return_value=mock_client):
            with patch("agent_trust_lab.sandbox.image.ImageManager") as mock_img_mgr:
                mock_img_mgr.return_value.ensure_image.return_value = True
                checker = CodeHalluChecker()
                report = checker.check("[].notexist()")
                assert report.hallucination_type == "naming"
                assert "AttributeError" in (report.error_message or "")
