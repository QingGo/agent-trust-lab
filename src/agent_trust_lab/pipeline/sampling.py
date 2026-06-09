"""Multi-sample classification for orchestrator pipeline.

Handles adaptive sampling and self-consistency via repeated GSAR
classifications with score averaging and standard deviation computation.
"""

from typing import Any, Dict, List

from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import HalluStepReport
from agent_trust_lab.models.trajectory import SecureTrajectory
from agent_trust_lab.pipeline.models import _std_dev

logger = get_logger("orchestrator")


def run_adaptive_sampling(
    hallucination_steps: List[HalluStepReport],
    trajectory: SecureTrajectory,
    classifier: Any,
    all_triples: List[Dict[str, Any]],
    config: Any,
) -> list[int]:
    """Run additional GSAR classifications for steps with high GSAR-NLI disagreement.

    Returns the list of step indices that were re-sampled (for re-applying
    faithfulness check by the caller).
    """
    threshold = config.adaptive_disagreement_threshold
    high_disagreement_indices = [
        h.step_index
        for h in hallucination_steps
        if h.gsar_nli_disagreement >= threshold
    ]
    if not high_disagreement_indices:
        return []

    max_samples = config.adaptive_max_samples
    logger.info(
        "Adaptive sampling triggered for %d/%d steps (disagreement >= %.2f)",
        len(high_disagreement_indices), len(hallucination_steps), threshold,
    )

    all_runs = multi_sample_classify(
        hallucination_steps, trajectory, classifier, all_triples,
        num_extra_runs=max(1, max_samples - 1),
        run_name="Adaptive sampling",
    )
    if len(all_runs) < 2:
        return []

    average_step_scores(
        hallucination_steps, all_runs, high_disagreement_indices,
        label_prefix=f"Adaptive resample ({len(all_runs)} runs)",
    )
    return high_disagreement_indices


def run_self_consistency(
    hallucination_steps: List[HalluStepReport],
    trajectory: SecureTrajectory,
    classifier: Any,
    all_triples: List[Dict[str, Any]],
    config: Any,
) -> None:
    """Run GSAR classification N times and average scores (self-consistency)."""
    n_samples = config.self_consistency_samples
    logger.info("Self-consistency: running %d classification rounds", n_samples)

    all_runs = multi_sample_classify(
        hallucination_steps, trajectory, classifier, all_triples,
        num_extra_runs=n_samples - 1,
        run_name="Self-consistency",
    )
    actual_runs = len(all_runs)
    if actual_runs < 2:
        return

    all_indices = list(range(len(hallucination_steps)))
    average_step_scores(
        hallucination_steps, all_runs, all_indices,
        label_prefix=f"SC ({actual_runs} runs)",
        compute_std=True,
    )


def multi_sample_classify(
    base_steps: List[HalluStepReport],
    trajectory: SecureTrajectory,
    classifier: Any,
    all_triples: List[Dict[str, Any]],
    num_extra_runs: int,
    run_name: str,
) -> List[List[HalluStepReport]]:
    """Run classifier N extra times and collect all results.

    Returns [base_run, run_1, run_2, ...] where base_run is the
    original classification. Failed runs are logged and skipped.
    """
    all_runs: List[List[HalluStepReport]] = [base_steps]
    for run_idx in range(num_extra_runs):
        try:
            run_result = classifier.classify(trajectory.steps, all_triples)
            if run_result:
                all_runs.append(run_result)
        except Exception as e:
            logger.warning(
                "%s run %d/%d failed: %s",
                run_name, run_idx + 1, num_extra_runs, e,
            )
    return all_runs


def average_step_scores(
    steps: List[HalluStepReport],
    all_runs: List[List[HalluStepReport]],
    step_indices: list,
    label_prefix: str = "",
    compute_std: bool = False,
) -> None:
    """Average G/U/C/F scores across multiple classification runs.

    Args:
        steps: Original step reports to update in-place.
        all_runs: [base_run, run_1, ...] from multi_sample_classify.
        step_indices: Which step indices to average (all or subset).
        label_prefix: Human-readable label for explanation text.
        compute_std: If True, compute and store standard deviations.
    """
    for step_idx in step_indices:
        step_reports = [
            run[step_idx] for run in all_runs if step_idx < len(run)
        ]
        if len(step_reports) < 2:
            continue

        g_mean = round(sum(r.g_score for r in step_reports) / len(step_reports), 4)
        u_mean = round(sum(r.u_score for r in step_reports) / len(step_reports), 4)
        c_mean = round(sum(r.c_score for r in step_reports) / len(step_reports), 4)
        f_mean = round(
            sum(r.faithfulness_score for r in step_reports) / len(step_reports), 4
        )

        original = steps[step_idx]
        original.g_score = g_mean
        original.u_score = u_mean
        original.c_score = c_mean
        original.faithfulness_score = f_mean

        if label_prefix:
            if compute_std:
                g_vals = [r.g_score for r in step_reports]
                u_vals = [r.u_score for r in step_reports]
                c_vals = [r.c_score for r in step_reports]
                f_vals = [r.faithfulness_score for r in step_reports]
                original.sc_samples = len(all_runs)
                original.sc_g_std = round(_std_dev(g_vals), 4)
                original.sc_u_std = round(_std_dev(u_vals), 4)
                original.sc_c_std = round(_std_dev(c_vals), 4)
                original.sc_f_std = round(_std_dev(f_vals), 4)
                original.explanation = (
                    f"{original.explanation} | "
                    f"{label_prefix}: "
                    f"g={g_mean}+/-{original.sc_g_std}, "
                    f"u={u_mean}+/-{original.sc_u_std}, "
                    f"c={c_mean}+/-{original.sc_c_std}, "
                    f"f={f_mean}+/-{original.sc_f_std}"
                )
            else:
                original.explanation = (
                    f"{original.explanation} | "
                    f"{label_prefix}: g={g_mean}, u={u_mean}, c={c_mean}, f={f_mean}"
                )
