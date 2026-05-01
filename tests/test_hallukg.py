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
    def test_default_construction(self):
        extractor = TripleExtractor()
        assert extractor.model_name == "gpt-4o-mini"

    def test_custom_model_construction(self):
        extractor = TripleExtractor(model_name="custom-model")
        assert extractor.model_name == "custom-model"

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
        assert result["score"] == 0.92

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
            assert r["score"] == 0.92

    def test_batch_anchor_empty_list(self):
        reasoner = AnchoringReasoner()
        results = reasoner.batch_anchor([])
        assert results == []


class TestGSARClassifier:
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
        assert labels == ["Grounded", "Grounded", "Complementary", "Grounded", "Grounded"]

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
    def test_default_construction(self):
        checker = CodeHalluChecker()
        assert checker.timeout == 30

    def test_custom_timeout_construction(self):
        checker = CodeHalluChecker(timeout=60)
        assert checker.timeout == 60

    def test_check_returns_code_hallu_report(self):
        checker = CodeHalluChecker()
        report = checker.check(
            code="import fake_lib",
            test_command="python test.py",
        )
        assert isinstance(report, CodeHalluReport)

    def test_check_default_hallucination_type(self):
        checker = CodeHalluChecker()
        report = checker.check("code", "cmd")
        assert report.hallucination_type == "naming"
        assert report.step_index == 0

    def test_check_truncates_code_snippet(self):
        checker = CodeHalluChecker()
        long_code = "x" * 200
        report = checker.check(long_code, "cmd")
        assert len(report.code_snippet) == 100

    def test_check_preserves_expected_error(self):
        checker = CodeHalluChecker()
        report = checker.check("code", "cmd", expected_error="ImportError")
        assert report.expected_error_pattern == "ImportError"

    def test_batch_check_filters_steps(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="thought", content="thinking"),
            TrajectoryStep(type="code_generation", content="code1"),
            TrajectoryStep(type="observation", content="result"),
            TrajectoryStep(type="trap_injection", content="trap1"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        assert len(reports) == 2
        assert reports[0].step_index == 1
        assert reports[1].step_index == 3

    def test_batch_check_cycles_hallucination_types(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="code_generation", content="c1"),
            TrajectoryStep(type="code_generation", content="c2"),
            TrajectoryStep(type="code_generation", content="c3"),
            TrajectoryStep(type="code_generation", content="c4"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        types = [r.hallucination_type for r in reports]
        assert types == ["mapping", "naming", "parameter", "logic_hallucination"]

    def test_batch_check_second_cycle(self):
        checker = CodeHalluChecker()
        steps = [
            TrajectoryStep(type="code_generation", content="c1"),
            TrajectoryStep(type="code_generation", content="c2"),
            TrajectoryStep(type="code_generation", content="c3"),
            TrajectoryStep(type="code_generation", content="c4"),
            TrajectoryStep(type="code_generation", content="c5"),
        ]
        trajectory = SecureTrajectory(steps=steps, security_events=[])
        reports = checker.batch_check(trajectory)
        types = [r.hallucination_type for r in reports]
        assert types == ["mapping", "naming", "parameter", "logic_hallucination", "mapping"]

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
        report = checker.check("code", "cmd")
        assert "correct package name" in report.fix_suggestion.lower()

    def test_check_has_error_message(self):
        checker = CodeHalluChecker()
        report = checker.check("code", "cmd")
        assert "ImportError" in report.error_message
