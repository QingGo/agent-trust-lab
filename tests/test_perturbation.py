from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep
from agent_trust_lab.perturbation import (
    PERTURBATIONS,
    PerturbationResult,
    PerturbationTester,
    _perturb_noise,
    _perturb_remove_middle,
    _perturb_reorder,
    _perturb_truncate,
)


def _make_traj(steps=None):
    if steps is None:
        steps = [
            TrajectoryStep(type="sandbox_init", content="init"),
            TrajectoryStep(type="thought", content="I should list files"),
            TrajectoryStep(type="action", content="ls /tmp"),
            TrajectoryStep(type="observation", content="file1 file2"),
            TrajectoryStep(type="thought", content="Good, I found files"),
            TrajectoryStep(type="trap_injection", content="injected"),
        ]
    return SecureTrajectory(
        steps=steps,
        security_events=[],
        policy_rules_applied=[],
        actual_violations=[],
        metadata={},
    )


class TestPerturbTruncate:
    def test_truncate_50(self):
        traj = _make_traj()
        perturbed = _perturb_truncate(traj, 0.5)
        assert len(perturbed.steps) == len(traj.steps)
        for orig, pert in zip(traj.steps, perturbed.steps):
            assert len(pert.content) <= len(orig.content)
            assert pert.type == orig.type

    def test_truncate_75(self):
        traj = _make_traj()
        perturbed = _perturb_truncate(traj, 0.75)
        assert len(perturbed.steps) == len(traj.steps)

    def test_truncate_preserves_non_content_fields(self):
        traj = _make_traj()
        perturbed = _perturb_truncate(traj, 0.5)
        assert perturbed.policy_rules_applied == []
        assert perturbed.actual_violations == []


class TestPerturbReorder:
    def test_reorder_preserves_step_count(self):
        traj = _make_traj()
        perturbed = _perturb_reorder(traj)
        assert len(perturbed.steps) == len(traj.steps)

    def test_reorder_does_not_affect_critical_types(self):
        traj = _make_traj()
        perturbed = _perturb_reorder(traj)
        for s in perturbed.steps:
            if s.type in ("trap_injection", "error", "sandbox_init"):
                pass

    def test_reorder_swaps_something(self):
        traj = _make_traj([
            TrajectoryStep(type="thought", content="A", tools_called=[]),
            TrajectoryStep(type="thought", content="B", tools_called=[]),
        ])
        perturbed = _perturb_reorder(traj)
        assert perturbed.steps[0].content == "B"
        assert perturbed.steps[1].content == "A"


class TestPerturbRemoveMiddle:
    def test_remove_middle_reduces_count(self):
        traj = _make_traj()
        perturbed = _perturb_remove_middle(traj)
        assert len(perturbed.steps) == len(traj.steps) - 1

    def test_remove_middle_keeps_critical(self):
        traj = _make_traj([
            TrajectoryStep(type="sandbox_init", content="init"),
            TrajectoryStep(type="trap_injection", content="inj"),
            TrajectoryStep(type="error", content="err"),
        ])
        perturbed = _perturb_remove_middle(traj)
        assert len(perturbed.steps) == len(traj.steps)


class TestPerturbNoise:
    def test_noise_adds_chars(self):
        traj = _make_traj()
        perturbed = _perturb_noise(traj, noise_chars=10)
        for orig, pert in zip(traj.steps, perturbed.steps):
            if orig.type not in ("trap_injection", "sandbox_init"):
                assert len(pert.content) > len(orig.content)

    def test_noise_does_not_affect_critical_types(self):
        traj = _make_traj()
        perturbed = _perturb_noise(traj, noise_chars=5)
        for orig, pert in zip(traj.steps, perturbed.steps):
            if orig.type in ("trap_injection", "sandbox_init"):
                assert pert.content == orig.content


class TestPerturbationRegistry:
    def test_all_registered(self):
        assert "truncate_50" in PERTURBATIONS
        assert "truncate_75" in PERTURBATIONS
        assert "reorder" in PERTURBATIONS
        assert "remove_middle" in PERTURBATIONS
        assert "noise" in PERTURBATIONS

    def test_each_callable(self):
        traj = _make_traj()
        for name, fn in PERTURBATIONS.items():
            result = fn(traj)
            assert isinstance(result, SecureTrajectory), f"{name} failed"


class TestPerturbationResult:
    def test_to_dict(self):
        pr = PerturbationResult(
            perturbation_name="truncate_50",
            original_scores={"avg_g_score": 0.8},
            perturbed_scores={"avg_g_score": 0.75},
            deltas={"avg_g_score": 0.05},
            max_delta=0.05,
            unstable=False,
        )
        d = pr.to_dict()
        assert d["perturbation"] == "truncate_50"
        assert d["max_delta"] == 0.05
        assert d["unstable"] is False

    def test_to_dict_with_unstable(self):
        pr = PerturbationResult(
            perturbation_name="reorder",
            original_scores={"avg_faithfulness": 0.9},
            perturbed_scores={"avg_faithfulness": 0.6},
            deltas={"avg_faithfulness": 0.3},
            max_delta=0.3,
            unstable=True,
        )
        d = pr.to_dict()
        assert d["unstable"] is True
        assert d["max_delta"] == 0.3


class TestPerturbationTester:
    def test_extract_summary_scores_empty(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.compliance = None
        result.hallucination_steps = []
        scores = PerturbationTester._extract_summary_scores(result)
        assert scores == {}

    def test_extract_summary_scores_with_compliance(self):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.compliance = MagicMock(critical_count=2, high_count=1)
        result.hallucination_steps = []
        scores = PerturbationTester._extract_summary_scores(result)
        assert scores["critical_count"] == 2.0
        assert scores["high_count"] == 1.0
