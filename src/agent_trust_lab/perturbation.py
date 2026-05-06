"""Trajectory perturbation functions for robustness testing.

Phase 3.4: Apply controlled perturbations to cached trajectories and
measure score stability. No harness re-execution needed.
"""

import copy
import random
from typing import Any, Dict, List, Optional

from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep


def _perturb_truncate(
    trajectory: SecureTrajectory, fraction: float
) -> SecureTrajectory:
    """Truncate each step's content to a fraction of its original length."""
    new_steps = []
    for step in trajectory.steps:
        trunc_len = max(int(len(step.content) * fraction), 1)
        new_steps.append(
            TrajectoryStep(
                type=step.type,
                content=step.content[:trunc_len],
                tools_called=list(step.tools_called),
                metadata=dict(step.metadata),
            )
        )
    return SecureTrajectory(
        steps=new_steps,
        security_events=copy.deepcopy(trajectory.security_events),
        dry_run_log=trajectory.dry_run_log,
        policy_rules_applied=list(trajectory.policy_rules_applied),
        actual_violations=list(trajectory.actual_violations),
        metadata=dict(trajectory.metadata),
    )


def _perturb_reorder(trajectory: SecureTrajectory) -> SecureTrajectory:
    """Swap adjacent pairs of non-critical steps."""
    steps = [
        TrajectoryStep(
            type=s.type,
            content=s.content,
            tools_called=list(s.tools_called),
            metadata=dict(s.metadata),
        )
        for s in trajectory.steps
    ]
    critical_types = {"trap_injection", "error", "sandbox_init"}
    i = len(steps) - 1
    while i >= 1:
        if steps[i].type not in critical_types and steps[i - 1].type not in critical_types:
            steps[i], steps[i - 1] = steps[i - 1], steps[i]
            i -= 2
        else:
            i -= 1
    return SecureTrajectory(
        steps=steps,
        security_events=copy.deepcopy(trajectory.security_events),
        dry_run_log=trajectory.dry_run_log,
        policy_rules_applied=list(trajectory.policy_rules_applied),
        actual_violations=list(trajectory.actual_violations),
        metadata=dict(trajectory.metadata),
    )


def _perturb_remove_middle(trajectory: SecureTrajectory) -> SecureTrajectory:
    """Remove one non-critical middle step."""
    steps = [
        TrajectoryStep(
            type=s.type,
            content=s.content,
            tools_called=list(s.tools_called),
            metadata=dict(s.metadata),
        )
        for s in trajectory.steps
    ]
    candidate_indices = [
        i
        for i, s in enumerate(steps)
        if s.type not in ("trap_injection", "error", "sandbox_init")
    ]
    if candidate_indices:
        mid_idx = candidate_indices[len(candidate_indices) // 2]
        steps.pop(mid_idx)
    return SecureTrajectory(
        steps=steps,
        security_events=copy.deepcopy(trajectory.security_events),
        dry_run_log=trajectory.dry_run_log,
        policy_rules_applied=list(trajectory.policy_rules_applied),
        actual_violations=list(trajectory.actual_violations),
        metadata=dict(trajectory.metadata),
    )


def _perturb_noise(
    trajectory: SecureTrajectory, noise_chars: int = 5
) -> SecureTrajectory:
    """Add random characters to non-critical step content."""
    new_steps = []
    for step in trajectory.steps:
        content = step.content
        if step.type not in ("trap_injection", "sandbox_init"):
            noise = "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=noise_chars))  # noqa: S311
            content = content + " " + noise
        new_steps.append(
            TrajectoryStep(
                type=step.type,
                content=content,
                tools_called=list(step.tools_called),
                metadata=dict(step.metadata),
            )
        )
    return SecureTrajectory(
        steps=new_steps,
        security_events=copy.deepcopy(trajectory.security_events),
        dry_run_log=trajectory.dry_run_log,
        policy_rules_applied=list(trajectory.policy_rules_applied),
        actual_violations=list(trajectory.actual_violations),
        metadata=dict(trajectory.metadata),
    )


PERTURBATIONS: Dict[str, Any] = {
    "truncate_50": lambda traj: _perturb_truncate(traj, 0.5),
    "truncate_75": lambda traj: _perturb_truncate(traj, 0.75),
    "reorder": _perturb_reorder,
    "remove_middle": _perturb_remove_middle,
    "noise": _perturb_noise,
}


class PerturbationResult:
    """Holds before/after comparison for one perturbation type."""

    def __init__(
        self,
        perturbation_name: str,
        original_scores: Optional[Dict[str, float]] = None,
        perturbed_scores: Optional[Dict[str, float]] = None,
        deltas: Optional[Dict[str, float]] = None,
        max_delta: float = 0.0,
        unstable: bool = False,
    ) -> None:
        self.perturbation_name = perturbation_name
        self.original_scores = original_scores or {}
        self.perturbed_scores = perturbed_scores or {}
        self.deltas = deltas or {}
        self.max_delta = max_delta
        self.unstable = unstable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation": self.perturbation_name,
            "original_scores": self.original_scores,
            "perturbed_scores": self.perturbed_scores,
            "deltas": self.deltas,
            "max_delta": self.max_delta,
            "unstable": self.unstable,
        }


class PerturbationTester:
    """Tests score stability by applying perturbations to trajectories.

    Takes a cached trajectory, applies each perturbation type, re-runs
    GSAR + faithfulness, and computes score deltas. No harness execution
    is performed - only the audit and hallukg pipelines are re-applied.

    Usage:
        tester = PerturbationTester(orchestrator)
        results = tester.run(trap, evaluation_result)
        for r in results:
            print(f"{r.perturbation_name}: max_delta={r.max_delta}")
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orch = orchestrator

    def run(
        self,
        trap: Any,
        original: Any,
        perturbation_names: Optional[List[str]] = None,
        stability_threshold: float = 0.15,
        seed: Optional[int] = None,
    ) -> List[PerturbationResult]:
        """Run perturbation tests and return comparison results.

        Args:
            trap: EnhancedTrapDef for audit context.
            original: EvaluationResult with the original trajectory and scores.
            perturbation_names: List of perturbation keys to test. Default: all.
            stability_threshold: Delta above which the result is flagged unstable.
            seed: Random seed for noise perturbation.

        Returns:
            List of PerturbationResult, one per perturbation type.
        """
        if seed is not None:
            random.seed(seed)

        names = perturbation_names or list(PERTURBATIONS.keys())
        results: List[PerturbationResult] = []

        original_scores = self._extract_summary_scores(original)

        for name in names:
            perturb_fn = PERTURBATIONS.get(name)
            if perturb_fn is None:
                continue

            try:
                perturbed_traj = perturb_fn(original.trajectory)
            except Exception as e:
                from agent_trust_lab.log import get_logger
                get_logger("perturbation").warning(
                    "Perturbation '%s' failed: %s", name, e
                )
                continue

            perturbed_result = self._orch.run_single(
                trap=trap,
                harness=None,
                mutate=False,
            )
            perturbed_result.trajectory = perturbed_traj
            perturbed_result.compliance = self._orch._audit_compliance(
                perturbed_traj,
                is_code_agent=trap.category == "code_agent",
                is_benign=trap.trap_type in ("benign_control", "benign_code_control"),
            )
            if not self._orch.config.skip_hallukg:
                perturbed_result.hallucination_steps, perturbed_result.code_agent_checks = (
                    self._orch._run_hallukg(
                        trajectory=perturbed_traj,
                        is_code_agent=trap.category == "code_agent",
                        knowledge_source=trap.knowledge_source or "",
                    )
                )

            perturbed_scores = self._extract_summary_scores(perturbed_result)
            deltas = {}
            max_delta = 0.0
            for key in original_scores:
                if key in perturbed_scores:
                    d = round(abs(original_scores[key] - perturbed_scores[key]), 4)
                    deltas[key] = d
                    if d > max_delta:
                        max_delta = d

            results.append(
                PerturbationResult(
                    perturbation_name=name,
                    original_scores=original_scores,
                    perturbed_scores=perturbed_scores,
                    deltas=deltas,
                    max_delta=max_delta,
                    unstable=max_delta >= stability_threshold,
                )
            )

        return results

    @staticmethod
    def _extract_summary_scores(result: Any) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if result.compliance is not None:
            scores["critical_count"] = float(result.compliance.critical_count)
            scores["high_count"] = float(result.compliance.high_count)
        if result.hallucination_steps:
            n = len(result.hallucination_steps)
            scores["avg_g_score"] = sum(h.g_score for h in result.hallucination_steps) / n
            scores["avg_u_score"] = sum(h.u_score for h in result.hallucination_steps) / n
            scores["avg_c_score"] = sum(h.c_score for h in result.hallucination_steps) / n
            scores["avg_faithfulness"] = (
                sum(h.faithfulness_score for h in result.hallucination_steps) / n
            )
        return scores
