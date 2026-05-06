"""Tests for calibration module: Platt scaling, Cohen's kappa, profiles, CLI."""

import json
import os
from unittest.mock import patch

import pytest

from agent_trust_lab.calibration.profile import (
    CalibrationProfile,
    _apply_calibration_to_results,
    build_distribution_signature,
    check_calibration_freshness,
    compute_cohens_kappa,
    list_profiles,
    load_profile,
    run_calibration,
    save_profile,
)
from agent_trust_lab.calibration.scaler import (
    apply_calibrated_score,
    fit_calibration,
    fit_platt_scaling,
)
from agent_trust_lab.report import ReportGenerator


def _make_step(**kwargs):
    """Create a step dict with default zero scores."""
    defaults = {
        "step_index": 0, "g_score": 0.0, "u_score": 0.0,
        "c_score": 0.0, "faithfulness_score": 0.0,
    }
    defaults.update(kwargs)
    return defaults


def _make_hallu_result(trap_id, trap_type, steps):
    """Create a result dict with hallucination steps."""
    return {
        "trap_id": trap_id,
        "trap_type": trap_type,
        "hallucination": {"steps": steps},
    }


class TestPlattScaling:
    def test_fit_and_apply_calibration(self):
        raw_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        binary_labels = [0, 0, 0, 1, 1]
        result = fit_platt_scaling(raw_scores, binary_labels)
        assert result is not None
        a, b = result

        cal_low = apply_calibrated_score(0.1, a, b)
        cal_high = apply_calibrated_score(0.9, a, b)
        assert cal_low < cal_high

    def test_fit_returns_none_for_identical_labels(self):
        result = fit_platt_scaling([0.1, 0.5, 0.9], [1, 1, 1])
        assert result is None

    def test_fit_returns_none_for_insufficient_samples(self):
        result = fit_platt_scaling([0.5], [1])
        assert result is None

    def test_fit_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_platt_scaling([0.1, 0.5], [1])

    def test_apply_calibrated_score_extremes(self):
        a, b = 2.0, -1.0
        cal_0 = apply_calibrated_score(0.0, a, b)
        cal_1 = apply_calibrated_score(1.0, a, b)
        assert 0.0 <= cal_0 <= 1.0
        assert 0.0 <= cal_1 <= 1.0
        assert cal_0 != cal_1

    def test_apply_calibrated_score_large_logit(self):
        a, b = 200.0, 0.0
        cal = apply_calibrated_score(0.8, a, b)
        assert cal == 0.0

    def test_apply_calibrated_score_negative_large_logit(self):
        a, b = -200.0, 0.0
        cal = apply_calibrated_score(0.8, a, b)
        assert cal == 1.0

    def test_fit_calibration_convenience(self):
        raw = [0.2, 0.4, 0.6, 0.8]
        human = [0.1, 0.3, 0.7, 0.9]
        result = fit_calibration(raw, human, threshold=0.5)
        assert result is not None
        a, b = result
        cal_low = apply_calibrated_score(0.2, a, b)
        cal_high = apply_calibrated_score(0.8, a, b)
        assert cal_low < cal_high

    def test_fit_platt_with_mixed_scores(self):
        raw_scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        binary_labels = [0, 0, 1, 1, 1]
        result = fit_platt_scaling(raw_scores, binary_labels)
        assert result is not None
        a, b = result
        cal_mid = apply_calibrated_score(0.5, a, b)
        assert 0.0 <= cal_mid <= 1.0

    def test_fit_with_cv_folds(self):
        raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        binary_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        result = fit_platt_scaling(raw_scores, binary_labels, cv_folds=3)
        assert result is not None
        a, b = result
        cal_low = apply_calibrated_score(0.2, a, b)
        cal_high = apply_calibrated_score(0.9, a, b)
        assert cal_low < cal_high

    def test_fit_cv_falls_back_on_small_samples(self):
        raw_scores = [0.2, 0.4, 0.6, 0.8]
        binary_labels = [0, 0, 1, 1]
        result = fit_platt_scaling(raw_scores, binary_labels, cv_folds=5)
        assert result is not None
        a, b = result

    def test_fit_cv_disabled_with_zero_folds(self):
        raw_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        binary_labels = [0, 0, 0, 1, 1]
        result = fit_platt_scaling(raw_scores, binary_labels, cv_folds=0)
        assert result is not None
        a, b = result

    def test_small_sample_warning(self, caplog):
        import logging

        from agent_trust_lab.calibration.scaler import logger as scaler_logger

        scaler_logger.setLevel(logging.WARNING)
        raw_scores = [0.1, 0.3, 0.5]
        binary_labels = [0, 0, 1]
        result = fit_platt_scaling(raw_scores, binary_labels, cv_folds=0)
        assert result is not None
        assert "may be unreliable" in caplog.text


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels_a = ["Grounded", "Ungrounded", "Grounded", "Contradicted"]
        labels_b = ["Grounded", "Ungrounded", "Grounded", "Contradicted"]
        kappa, ci = compute_cohens_kappa(labels_a, labels_b)
        assert kappa == 1.0
        assert ci == (1.0, 1.0)

    def test_complete_disagreement(self):
        labels_a = ["Grounded", "Ungrounded", "Ungrounded"]
        labels_b = ["Ungrounded", "Grounded", "Grounded"]
        kappa, ci = compute_cohens_kappa(labels_a, labels_b)
        assert kappa < 0.0

    def test_random_agreement(self):
        labels_a = ["Grounded", "Ungrounded", "Grounded", "Complementary"]
        labels_b = ["Ungrounded", "Grounded", "Ungrounded", "Grounded"]
        kappa, ci = compute_cohens_kappa(labels_a, labels_b)
        assert -1.0 <= kappa <= 0.5

    def test_partial_agreement(self):
        labels_a = ["Grounded", "Grounded", "Ungrounded", "Ungrounded", "Grounded", "Ungrounded"]
        labels_b = ["Grounded", "Grounded", "Grounded", "Ungrounded", "Grounded", "Ungrounded"]
        kappa, ci = compute_cohens_kappa(labels_a, labels_b)
        assert 0.0 < kappa < 1.0

    def test_confidence_interval_range(self):
        labels_a = ["Grounded"] * 50 + ["Ungrounded"] * 50
        labels_b = ["Grounded"] * 45 + ["Ungrounded"] * 5 + ["Grounded"] * 5 + ["Ungrounded"] * 45
        kappa, (ci_low, ci_high) = compute_cohens_kappa(labels_a, labels_b)
        assert -1.0 <= ci_low <= kappa <= ci_high <= 1.0

    def test_empty_labels(self):
        kappa, ci = compute_cohens_kappa([], [])
        assert kappa == 0.0

    def test_single_label(self):
        kappa, ci = compute_cohens_kappa(["Grounded"], ["Grounded"])
        assert kappa == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_cohens_kappa(["Grounded"], ["Grounded", "Ungrounded"])

    def test_unified_categories(self):
        labels_a = ["Grounded", "Complementary"]
        labels_b = ["Ungrounded", "Contradicted"]
        kappa, ci = compute_cohens_kappa(labels_a, labels_b)
        assert -1.0 <= kappa <= 0.0

    def test_large_sample_kappa(self):
        labels_a = ["Grounded"] * 100 + ["Ungrounded"] * 100
        labels_b = ["Grounded"] * 90 + ["Ungrounded"] * 10 + ["Grounded"] * 10 + ["Ungrounded"] * 90
        kappa, (ci_low, ci_high) = compute_cohens_kappa(labels_a, labels_b)
        assert 0.0 < kappa < 1.0
        assert ci_low < kappa < ci_high


class TestCalibrationProfile:
    def test_roundtrip_dict(self):
        profile = CalibrationProfile(
            profile_id="test-v1",
            benchmark="test-bench",
            version="1.0",
            created_at="2026-01-01T00:00:00 UTC",
            sample_count=100,
            kappa_gsar=0.72,
            kappa_gsar_ci=(0.65, 0.79),
            kappa_compliance=0.68,
            platt_params={
                "g_score": {"A": -2.0, "B": 1.5},
            },
        )
        data = profile.to_dict()
        restored = CalibrationProfile.from_dict(data)
        assert restored.profile_id == "test-v1"
        assert restored.kappa_gsar == 0.72
        assert restored.kappa_gsar_ci == (0.65, 0.79)
        assert restored.platt_params["g_score"]["A"] == -2.0

    def test_default_values(self):
        profile = CalibrationProfile(profile_id="min", benchmark="b", version="v")
        assert profile.sample_count == 0
        assert profile.kappa_gsar == 0.0
        assert profile.platt_params == {}

    def test_get_calibrated_score(self):
        profile = CalibrationProfile(
            profile_id="test",
            benchmark="b",
            version="v",
            platt_params={
                "g_score": {"A": -2.0, "B": 1.0},
            },
        )
        cal = profile.get_calibrated_score("g_score", 0.5)
        assert cal is not None
        assert 0.0 <= cal <= 1.0

    def test_get_calibrated_score_missing(self):
        profile = CalibrationProfile(profile_id="test", benchmark="b", version="v")
        assert profile.get_calibrated_score("g_score", 0.5) is None


class TestProfileStorage:
    def test_save_and_load_profile(self, tmp_path):
        calibration_dir = str(tmp_path / "calibration")
        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            profile = CalibrationProfile(
                profile_id="save-test",
                benchmark="test-bench",
                version="1.0",
                sample_count=50,
                kappa_gsar=0.85,
                kappa_gsar_ci=(0.80, 0.90),
                platt_params={"g_score": {"A": -1.5, "B": 0.8}},
            )
            path = save_profile(profile)
            assert os.path.isfile(path)

            loaded = load_profile("save-test")
            assert loaded is not None
            assert loaded.profile_id == "save-test"
            assert loaded.kappa_gsar == 0.85
            assert loaded.platt_params["g_score"]["A"] == -1.5

    def test_load_nonexistent_profile(self):
        loaded = load_profile("nonexistent-profile-xyz")
        assert loaded is None

    def test_list_profiles(self, tmp_path):
        calibration_dir = str(tmp_path / "calibration")
        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            assert list_profiles() == []
            save_profile(CalibrationProfile(profile_id="p1", benchmark="b1", version="v1"))
            save_profile(CalibrationProfile(profile_id="p2", benchmark="b2", version="v2"))
            profiles = list_profiles()
            assert sorted(profiles) == ["p1", "p2"]

    def test_list_profiles_empty_dir(self, tmp_path):
        calibration_dir = str(tmp_path / "nonexistent")
        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            assert list_profiles() == []


class TestApplyCalibrationToResults:
    def test_apply_adds_calibrated_fields(self):
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "hallucination": {
                        "step_count": 2,
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.8,
                                "u_score": 0.1,
                                "c_score": 0.05,
                                "faithfulness_score": 0.9,
                            },
                            {
                                "step_index": 1,
                                "gsar_label": "Ungrounded",
                                "g_score": 0.2,
                                "u_score": 0.7,
                                "c_score": 0.3,
                                "faithfulness_score": 0.4,
                            },
                        ],
                    },
                }
            ],
        }
        profile = CalibrationProfile(
            profile_id="apply-test",
            benchmark="b",
            version="v",
            platt_params={
                "g_score": {"A": -2.0, "B": 1.0},
                "faithfulness_score": {"A": -1.0, "B": 0.5},
            },
        )
        result = _apply_calibration_to_results(data, profile)
        steps = result["results"][0]["hallucination"]["steps"]
        assert "calibrated_g_score" in steps[0]
        assert "calibrated_faithfulness_score" in steps[0]
        assert 0.0 <= steps[0]["calibrated_g_score"] <= 1.0

    def test_apply_no_hallucination(self):
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [{"trap_id": "t1"}],
        }
        profile = CalibrationProfile(
            profile_id="test",
            benchmark="b",
            version="v",
            platt_params={"g_score": {"A": -1.0, "B": 0.0}},
        )
        result = _apply_calibration_to_results(data, profile)
        assert result["results"][0]["trap_id"] == "t1"


class TestRunCalibration:
    def test_run_calibration_creates_profile(self, tmp_path):
        results_path = str(tmp_path / "results.json")
        annotations_path = str(tmp_path / "annotations.json")
        calibration_dir = str(tmp_path / "calibration")

        results_data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "hallucination": {
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.8,
                                "u_score": 0.1,
                                "c_score": 0.05,
                                "faithfulness_score": 0.9,
                            },
                            {
                                "step_index": 1,
                                "gsar_label": "Ungrounded",
                                "g_score": 0.2,
                                "u_score": 0.7,
                                "c_score": 0.3,
                                "faithfulness_score": 0.4,
                            },
                        ],
                    },
                }
            ],
        }
        annotations_data = {
            "benchmark": "test-bench",
            "version": "1.0",
            "annotations": [
                {
                    "trap_id": "t1",
                    "step_index": 0,
                    "gsar_label": "Grounded",
                    "g_score": 0.9,
                    "faithfulness_score": 0.95,
                    "u_score": 0.1,
                    "c_score": 0.1,
                },
                {
                    "trap_id": "t1",
                    "step_index": 1,
                    "gsar_label": "Ungrounded",
                    "g_score": 0.1,
                    "faithfulness_score": 0.3,
                    "u_score": 0.8,
                    "c_score": 0.4,
                },
            ],
        }

        with open(results_path, "w") as f:
            json.dump(results_data, f)
        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f)

        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            profile = run_calibration(results_path, annotations_path, profile_id="e2e-test")
            assert profile.profile_id == "e2e-test"
            assert profile.sample_count == 2
            assert profile.benchmark == "test-bench"
            assert profile.kappa_gsar is not None
            assert os.path.isfile(os.path.join(calibration_dir, "e2e-test.json"))

    def test_run_calibration_no_matches(self, tmp_path):
        results_path = str(tmp_path / "no_match_results.json")
        annotations_path = str(tmp_path / "no_match_annotations.json")
        calibration_dir = str(tmp_path / "calibration")

        results_data = {
            "config": {"model": "test"},
            "results": [
                {
                    "trap_id": "t1",
                    "hallucination": {"steps": [{"step_index": 0, "gsar_label": "Grounded"}]},
                },
            ],
        }
        annotations_data = {
            "annotations": [
                {"trap_id": "t2", "step_index": 0, "gsar_label": "Grounded"},
            ],
        }
        with open(results_path, "w") as f:
            json.dump(results_data, f)
        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f)

        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            with pytest.raises(ValueError, match="No matching"):
                run_calibration(results_path, annotations_path)

    def test_run_calibration_no_hallucination_steps(self, tmp_path):
        results_path = str(tmp_path / "no_hallu.json")
        annotations_path = str(tmp_path / "ann.json")
        calibration_dir = str(tmp_path / "calibration")

        results_data = {
            "config": {"model": "test"},
            "results": [{"trap_id": "t1"}],
        }
        annotations_data = {
            "annotations": [
                {"trap_id": "t1", "step_index": 0, "gsar_label": "Grounded"},
            ],
        }
        with open(results_path, "w") as f:
            json.dump(results_data, f)
        with open(annotations_path, "w") as f:
            json.dump(annotations_data, f)

        with patch("agent_trust_lab.calibration.profile.DEFAULT_CALIBRATION_DIR", calibration_dir):
            with pytest.raises(ValueError, match="No matching"):
                run_calibration(results_path, annotations_path)


class TestReportWithCalibration:
    def test_report_generate_with_calibration(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test-model", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_01",
                    "trap_type": "parameter_hallucination",
                    "category": "general_agent",
                    "steps_count": 5,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                    "hallucination": {
                        "step_count": 2,
                        "avg_g_score": 0.5,
                        "avg_faithfulness": 0.6,
                        "labels": ["Grounded", "Ungrounded"],
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.8,
                                "u_score": 0.1,
                                "c_score": 0.05,
                                "faithfulness_score": 0.9,
                                "calibrated_g_score": 0.85,
                                "calibrated_faithfulness_score": 0.92,
                            },
                            {
                                "step_index": 1,
                                "gsar_label": "Ungrounded",
                                "g_score": 0.2,
                                "u_score": 0.7,
                                "c_score": 0.3,
                                "faithfulness_score": 0.3,
                                "calibrated_g_score": 0.15,
                                "calibrated_faithfulness_score": 0.25,
                            },
                        ],
                    },
                }
            ],
        }
        cal = {
            "profile_id": "test-profile",
            "kappa_gsar": 0.85,
        }
        html = generator.generate(data, calibration=cal)
        assert "Calibrated" in html
        assert "test-profile" in html
        assert "0.85" in html

    def test_report_generate_without_calibration(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "hallucination": {
                        "step_count": 1,
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.7,
                                "u_score": 0.1,
                                "c_score": 0.05,
                                "faithfulness_score": 0.8,
                            },
                        ],
                    },
                }
            ],
        }
        html = generator.generate(data)
        assert "Calibrated" not in html
        assert "test-profile" not in html


class TestCLICalibrate:
    def test_calibrate_list_empty(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        with patch("agent_trust_lab.calibration.profile.list_profiles", return_value=[]):
            result = runner.invoke(app, ["calibrate", "dummy.json", "--list"])
            assert result.exit_code == 0
            assert "No calibration profiles found" in result.stdout

    def test_calibrate_list_with_profiles(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        profile = CalibrationProfile(
            profile_id="cli-test",
            benchmark="test-bench",
            version="1.0",
            sample_count=42,
            kappa_gsar=0.75,
        )
        with (
            patch(
                "agent_trust_lab.calibration.profile.list_profiles",
                return_value=["cli-test"],
            ),
            patch(
                "agent_trust_lab.calibration.profile.load_profile",
                return_value=profile,
            ),
        ):
            result = runner.invoke(app, ["calibrate", "dummy.json", "--list"])
            assert result.exit_code == 0
            assert "cli-test" in result.stdout
            assert "0.750" in result.stdout

    def test_calibrate_missing_annotations_and_output(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["calibrate", "dummy.json"])
        assert result.exit_code == 1
        assert "Specify --annotations" in result.stdout or "--annotations" in result.stdout

    def test_calibrate_output_without_profile(self, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        results_path = str(tmp_path / "results.json")
        output_path = str(tmp_path / "calibrated.json")

        data = {"config": {"model": "test"}, "results": []}
        with open(results_path, "w") as f:
            json.dump(data, f)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "calibrate",
                results_path,
                "--output",
                output_path,
                "--profile-id",
                "nonexistent",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestDistributionSignature:
    def test_build_signature_from_results(self):
        data = {
            "results": [
                _make_hallu_result("t1", "authority_appeal", [
                    _make_step(
                        step_index=0, g_score=0.8, u_score=0.1,
                        c_score=0.05, faithfulness_score=0.9,
                    ),
                    _make_step(
                        step_index=1, g_score=0.2, u_score=0.7,
                        c_score=0.3, faithfulness_score=0.4,
                    ),
                ]),
            ],
        }
        sig = build_distribution_signature(data)
        assert "authority_appeal" in sig
        scores = sig["authority_appeal"]["scores"]
        assert scores["g_score"]["mean"] > 0.0
        assert scores["g_score"]["count"] == 2
        assert "std" in scores["g_score"]

    def test_build_signature_empty(self):
        sig = build_distribution_signature({"results": []})
        assert sig == {}

    def test_build_signature_no_hallucination(self):
        sig = build_distribution_signature({"results": [{"trap_id": "t1", "trap_type": "test"}]})
        assert sig == {}

    def test_build_signature_single_step_per_type(self):
        data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "benign_control",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.9,
                                                                                                                                                    u_score=0.0,
                                                                                                                                                    c_score=0.0,
                                                                                                                                                    faithfulness_score=0.95,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(data)
        scores = sig["benign_control"]["scores"]
        assert scores["g_score"]["mean"] == 0.9
        assert scores["g_score"]["std"] == 0.0
        assert scores["g_score"]["count"] == 1

    def test_build_signature_multiple_types(self):
        data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "type_a",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.5,
                                                                                                                                                    u_score=0.5,
                                                                                                                                                    c_score=0.0,
                                                                                                                                                    faithfulness_score=0.6,
                                ),
                        ],
                    },
                },
                {
                    "trap_id": "t2",
                    "trap_type": "type_b",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.8,
                                                                                                                                                    u_score=0.1,
                                                                                                                                                    c_score=0.1,
                                                                                                                                                    faithfulness_score=0.9,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(data)
        assert "type_a" in sig
        assert "type_b" in sig


class TestCalibrationFreshness:
    def test_fresh_when_similar_distribution(self):
        data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.8,
                                                                                                                                                    u_score=0.1,
                                                                                                                                                    c_score=0.05,
                                                                                                                                                    faithfulness_score=0.9,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(data)
        profile = CalibrationProfile(
            profile_id="fresh-test",
            benchmark="b",
            version="v",
            distribution_signature=sig,
        )
        is_fresh, drift, msg = check_calibration_freshness(profile, data)
        assert is_fresh is True
        assert drift == 0.0

    def test_stale_when_distribution_shifted(self):
        calib_data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.9,
                                                                                                                                                    u_score=0.0,
                                                                                                                                                    c_score=0.0,
                                                                                                                                                    faithfulness_score=0.95,
                                ),
                        ],
                    },
                },
            ],
        }
        current_data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.2,
                                                                                                                                                    u_score=0.8,
                                                                                                                                                    c_score=0.3,
                                                                                                                                                    faithfulness_score=0.3,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(calib_data)
        profile = CalibrationProfile(
            profile_id="stale-test",
            benchmark="b",
            version="v",
            distribution_signature=sig,
        )
        is_fresh, drift, msg = check_calibration_freshness(profile, current_data)
        assert drift > 0.3

    def test_fresh_when_no_signature(self):
        profile = CalibrationProfile(
            profile_id="no-sig",
            benchmark="b",
            version="v",
            distribution_signature={},
        )
        is_fresh, drift, msg = check_calibration_freshness(profile, {"results": []})
        assert is_fresh is True
        assert "No distribution signature" in msg

    def test_drift_score_reasonable(self):
        calib_data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.5,
                                                                                                                                                    u_score=0.3,
                                                                                                                                                    c_score=0.1,
                                                                                                                                                    faithfulness_score=0.6,
                                ),
                            _make_step(
                                                                                                                                                    step_index=1,
                                                                                                                                                    g_score=0.6,
                                                                                                                                                    u_score=0.2,
                                                                                                                                                    c_score=0.1,
                                                                                                                                                    faithfulness_score=0.7,
                                ),
                        ],
                    },
                },
            ],
        }
        current_data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.4,
                                                                                                                                                    u_score=0.4,
                                                                                                                                                    c_score=0.2,
                                                                                                                                                    faithfulness_score=0.5,
                                ),
                            _make_step(
                                                                                                                                                    step_index=1,
                                                                                                                                                    g_score=0.5,
                                                                                                                                                    u_score=0.3,
                                                                                                                                                    c_score=0.1,
                                                                                                                                                    faithfulness_score=0.6,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(calib_data)
        profile = CalibrationProfile(
            profile_id="drift-test",
            benchmark="b",
            version="v",
            distribution_signature=sig,
        )
        is_fresh, drift, msg = check_calibration_freshness(
            profile, current_data, drift_threshold=0.1
        )
        assert 0.0 < drift < 1.0

    def test_empty_results_returns_true(self):
        calib_data = {
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "test_type",
                    "hallucination": {
                        "steps": [
                            _make_step(
                                                                                                                                                    step_index=0,
                                                                                                                                                    g_score=0.5,
                                                                                                                                                    u_score=0.3,
                                                                                                                                                    c_score=0.1,
                                                                                                                                                    faithfulness_score=0.6,
                                ),
                        ],
                    },
                },
            ],
        }
        sig = build_distribution_signature(calib_data)
        profile = CalibrationProfile(
            profile_id="empty-res",
            benchmark="b",
            version="v",
            distribution_signature=sig,
        )
        is_fresh, drift, msg = check_calibration_freshness(
            profile, {"results": [{"trap_id": "t1"}]}
        )
        assert is_fresh is True

    def test_roundtrip_profile_with_signature(self):
        sig = {"test_type": {"scores": {"g_score": {"mean": 0.7, "std": 0.1, "count": 10}}}}
        profile = CalibrationProfile(
            profile_id="sig-test",
            benchmark="b",
            version="v",
            distribution_signature=sig,
        )
        data = profile.to_dict()
        assert "distribution_signature" in data
        restored = CalibrationProfile.from_dict(data)
        assert restored.distribution_signature == sig
