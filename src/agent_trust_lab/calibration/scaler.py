"""Platt scaling for calibrating model scores to human-aligned probabilities."""

import math
from typing import List, Optional, Tuple, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from agent_trust_lab.log import get_logger

logger = get_logger("calibration.scaler")


def fit_platt_scaling(
    raw_scores: List[float],
    binary_labels: List[int],
) -> Optional[Tuple[float, float]]:
    """Fit Platt scaling parameters using logistic regression.

    P(label=1 | score) = 1 / (1 + exp(a * score + b))

    Args:
        raw_scores: Model-produced scores in [0, 1].
        binary_labels: Human binary labels (0 = negative, 1 = positive).

    Returns:
        (a, b) tuple, or None if fitting fails (e.g., all labels identical).
    """
    if len(raw_scores) != len(binary_labels):
        raise ValueError(
            f"Length mismatch: {len(raw_scores)} scores vs {len(binary_labels)} labels"
        )
    if len(raw_scores) < 2:
        logger.warning("Insufficient samples for Platt scaling (need >= 2)")
        return None

    unique_labels = set(binary_labels)
    if len(unique_labels) < 2:
        logger.warning(
            "All labels are identical (%d), cannot fit Platt scaling", next(iter(unique_labels))
        )
        return None

    x_arr = np.array([[s] for s in raw_scores], dtype=float)
    y_arr = np.array(binary_labels, dtype=int)

    try:
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(x_arr, y_arr)
    except Exception as e:
        logger.error("Platt scaling fit failed: %s", e)
        return None

    coef = clf.coef_
    if coef is None:
        return None
    coef_arr: np.ndarray = cast(np.ndarray, coef)
    if coef_arr.shape[1] < 1:
        return None
    intercept = clf.intercept_
    if intercept is None:
        return None
    intercept_arr: np.ndarray = cast(np.ndarray, intercept)
    a_val = -float(coef_arr[0][0])
    b_val = -float(intercept_arr[0])
    logger.debug("Platt params: a=%.4f b=%.4f (n=%d)", a_val, b_val, len(raw_scores))
    return a_val, b_val


def apply_calibrated_score(raw_score: float, a: float, b: float) -> float:
    """Apply Platt scaling to convert a raw score to a calibrated probability.

    calibrated = 1 / (1 + exp(A * raw_score + B))

    Args:
        raw_score: Raw model score in [0, 1].
        a: Platt scaling slope parameter.
        b: Platt scaling intercept parameter.

    Returns:
        Calibrated probability in [0, 1].
    """
    logit = a * raw_score + b
    if logit > 100:
        return 0.0
    if logit < -100:
        return 1.0
    return 1.0 / (1.0 + math.exp(logit))


def fit_calibration(
    raw_scores: List[float],
    human_labels: List[float],
    threshold: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """Convenience: fit Platt scaling from continuous human confidence labels.

    Binarizes human labels at the given threshold, then fits Platt scaling.

    Args:
        raw_scores: Model-produced scores in [0, 1].
        human_labels: Human confidence scores in [0, 1].
        threshold: Threshold for binarizing human labels.

    Returns:
        (A, B) tuple, or None if fitting fails.
    """
    binary = [1 if label >= threshold else 0 for label in human_labels]
    return fit_platt_scaling(raw_scores, binary)
