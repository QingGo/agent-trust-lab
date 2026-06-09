"""Unit tests for pipeline/ package — pure logic classes and functions."""

import pytest

from agent_trust_lab.models.report import ComplianceReport, HalluStepReport
from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep
from agent_trust_lab.pipeline.models import EvaluationResult, _std_dev
from agent_trust_lab.pipeline.sampling import average_step_scores


class TestStdDev:
    def test_empty_returns_zero(self):
        assert _std_dev([]) == 0.0

    def test_single_value_returns_zero(self):
        assert _std_dev([5.0]) == 0.0

    def test_identical_values_returns_zero(self):
        assert _std_dev([3.0, 3.0, 3.0]) == 0.0

    def test_two_values(self):
        result = _std_dev([0.0, 1.0])
        assert round(result, 4) == 0.7071

    def test_three_values(self):
        result = _std_dev([2.0, 4.0, 6.0])
        assert round(result, 4) == 2.0


class TestAverageStepScores:
    def _make_report(self, index, g, u, c, f, label="Grounded"):
        return HalluStepReport(
            step_index=index,
            gsar_label=label,
            g_score=g,
            u_score=u,
            c_score=c,
            faithfulness_score=f,
            explanation=f"step{index}",
        )

    def test_averages_across_two_runs(self):
        base = [self._make_report(0, 0.8, 0.1, 0.0, 0.9)]
        run2 = [self._make_report(0, 0.6, 0.2, 0.1, 0.7)]
        average_step_scores(base, [base, run2], [0], label_prefix="avg")

        assert base[0].g_score == 0.7
        assert base[0].u_score == 0.15
        assert base[0].c_score == 0.05
        assert base[0].faithfulness_score == 0.8
        assert "g=0.7" in base[0].explanation

    def test_averages_three_runs(self):
        base = [self._make_report(0, 1.0, 0.0, 0.0, 1.0)]
        r2 = [self._make_report(0, 0.5, 0.2, 0.0, 0.5)]
        r3 = [self._make_report(0, 0.0, 0.5, 0.0, 0.0)]
        average_step_scores(base, [base, r2, r3], [0])

        assert base[0].g_score == 0.5
        assert round(base[0].u_score, 4) == round(0.7 / 3, 4)

    def test_single_run_skipped(self):
        base = [self._make_report(0, 0.9, 0.0, 0.0, 0.9)]
        original_g = base[0].g_score
        average_step_scores(base, [base], [0])
        assert base[0].g_score == original_g

    def test_with_std_dev(self):
        base = [self._make_report(0, 1.0, 0.0, 0.0, 1.0)]
        r2 = [self._make_report(0, 0.5, 0.2, 0.1, 0.7)]
        r3 = [self._make_report(0, 0.0, 0.5, 0.2, 0.5)]
        average_step_scores(
            base, [base, r2, r3], [0],
            label_prefix="SC (3 runs)", compute_std=True,
        )

        assert base[0].sc_samples == 3
        assert base[0].sc_g_std > 0
        assert base[0].sc_u_std > 0
        assert "g=" in base[0].explanation and "+/-" in base[0].explanation

    def test_subset_indices(self):
        base = [
            self._make_report(0, 0.8, 0.1, 0.0, 0.9),
            self._make_report(1, 0.3, 0.6, 0.0, 0.4),
        ]
        r2 = [
            self._make_report(0, 0.6, 0.2, 0.0, 0.8),
            self._make_report(1, 0.1, 0.8, 0.0, 0.2),
        ]
        average_step_scores(base, [base, r2], [0])

        assert base[0].g_score == 0.7
        assert base[1].g_score == 0.3

    def test_missing_step_in_run_skipped(self):
        base = [self._make_report(0, 0.9, 0.0, 0.0, 0.9)]
        short_run = []  # no steps
        average_step_scores(base, [base, short_run], [0])
        assert base[0].g_score == 0.9  # unchanged


class TestEvaluationResult:
    def _make_trajectory(self):
        return SecureTrajectory(
            steps=[
                TrajectoryStep(type="thought", content="thinking..."),
                TrajectoryStep(type="action", content="called tool"),
            ],
            security_events=[],
            dry_run_log="",
            policy_rules_applied=["rule1"],
            actual_violations=[],
            metadata={},
        )

    def test_summary_basic(self):
        result = EvaluationResult(
            trap_id="test_01",
            trap_type="tool_bypass",
            category="general_agent",
            trajectory=self._make_trajectory(),
            metadata={"difficulty": "hard"},
        )
        summary = result.summary()
        assert summary["trap_id"] == "test_01"
        assert summary["trap_type"] == "tool_bypass"
        assert summary["difficulty"] == "hard"
        assert summary["steps_count"] == 2
        assert summary["security_events"] == 0
        assert "compliance" not in summary
        assert "hallucination" not in summary

    def test_summary_with_compliance(self):
        result = EvaluationResult(
            trap_id="test_01",
            trap_type="tool_bypass",
            category="general_agent",
            trajectory=self._make_trajectory(),
            compliance=ComplianceReport(
                dimensions={"tool_auth": "pass"},
                critical_count=0,
                high_count=1,
                evidence=[],
            ),
        )
        summary = result.summary()
        assert summary["compliance"]["critical_count"] == 0
        assert summary["compliance"]["high_count"] == 1

    def test_summary_with_hallucination(self):
        result = EvaluationResult(
            trap_id="test_01",
            trap_type="tool_bypass",
            category="general_agent",
            trajectory=self._make_trajectory(),
            hallucination_steps=[
                HalluStepReport(
                    step_index=0, gsar_label="Grounded", g_score=0.9,
                    u_score=0.05, c_score=0.0, faithfulness_score=0.95,
                ),
                HalluStepReport(
                    step_index=1, gsar_label="Ungrounded", g_score=0.1,
                    u_score=0.9, c_score=0.0, faithfulness_score=0.3,
                ),
            ],
        )
        summary = result.summary()
        h = summary["hallucination"]
        assert h["step_count"] == 2
        assert h["avg_g_score"] == 0.5
        assert h["labels"] == ["Grounded", "Ungrounded"]

    def test_summary_with_error(self):
        result = EvaluationResult(
            trap_id="test_01",
            trap_type="tool_bypass",
            category="general_agent",
            trajectory=self._make_trajectory(),
            error="Something went wrong",
        )
        summary = result.summary()
        assert summary["error"] == "Something went wrong"

    def test_summary_empty_hallucination_skipped(self):
        result = EvaluationResult(
            trap_id="test_01",
            trap_type="tool_bypass",
            category="general_agent",
            trajectory=self._make_trajectory(),
            hallucination_steps=[],
        )
        summary = result.summary()
        assert "hallucination" not in summary

    def test_to_dict_from_dict_roundtrip(self):
        traj = self._make_trajectory()
        original = EvaluationResult(
            trap_id="rt_01",
            trap_type="backdoor",
            category="general_agent",
            trajectory=traj,
            mutated=True,
            mutation_seed=42,
            metadata={"key": "val"},
            hallucination_steps=[
                HalluStepReport(
                    step_index=0, gsar_label="Grounded", g_score=0.8,
                    u_score=0.1, c_score=0.0, faithfulness_score=0.9,
                    evidence=["proof"], explanation="looks good",
                ),
            ],
        )
        restored = EvaluationResult.from_dict(original.to_dict())
        assert restored.trap_id == "rt_01"
        assert restored.trap_type == "backdoor"
        assert restored.mutated is True
        assert restored.mutation_seed == 42
        assert restored.metadata == {"key": "val"}
        assert len(restored.hallucination_steps) == 1
        assert restored.hallucination_steps[0].gsar_label == "Grounded"
        assert restored.hallucination_steps[0].g_score == 0.8

    def test_from_dict_minimal(self):
        traj = self._make_trajectory()
        data = {
            "trap_id": "min",
            "trap_type": "test",
            "category": "general",
            "trajectory": traj.to_dict(),
        }
        result = EvaluationResult.from_dict(data)
        assert result.trap_id == "min"
        assert result.compliance is None
        assert result.hallucination_steps == []
