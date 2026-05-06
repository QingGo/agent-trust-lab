"""Platt scaling for calibrating model scores to human-aligned probabilities."""

import math
from typing import List, Optional, Tuple, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from agent_trust_lab.log import get_logger

logger = get_logger("calibration.scaler")

_MIN_SAMPLES_WARNING = 100
_MIN_SAMPLES_CV = 10


def fit_platt_scaling(
    raw_scores: List[float],
    binary_labels: List[int],
    cv_folds: int = 5,
) -> Optional[Tuple[float, float]]:
    """Fit Platt scaling parameters using logistic regression with cross-validation.

    P(label=1 | score) = 1 / (1 + exp(a * score + b))

    When sample count is sufficient, uses k-fold cross-validation to produce
    more reliable parameters and warns about cross-fold variance.

    Args:
        raw_scores: Model-produced scores in [0, 1].
        binary_labels: Human binary labels (0 = negative, 1 = positive).
        cv_folds: Number of cross-validation folds. Use 0 to disable CV and
            fit on all data. Default 5.

    Returns:
        (a, b) tuple, or None if fitting fails (e.g., all labels identical).
    """
    n = len(raw_scores)
    if n != len(binary_labels):
        raise ValueError(
            f"Length mismatch: {n} scores vs {len(binary_labels)} labels"
        )
    if n < 2:
        logger.warning("Insufficient samples for Platt scaling (need >= 2)")
        return None

    unique_labels = set(binary_labels)
    if len(unique_labels) < 2:
        logger.warning(
            "All labels are identical (%d), cannot fit Platt scaling", next(iter(unique_labels))
        )
        return None

    if n < _MIN_SAMPLES_WARNING:
        logger.warning(
            "Platt scaling with %d samples may be unreliable. "
            "Consider collecting >= %d samples or using isotonic regression. "
            "See sklearn.calibration.CalibratedClassifierCV docs.",
            n,
            _MIN_SAMPLES_WARNING,
        )

    x_arr = np.array([[s] for s in raw_scores], dtype=float)
    y_arr = np.array(binary_labels, dtype=int)

    actual_folds = cv_folds if cv_folds > 0 else 1
    if actual_folds > 1 and n >= _MIN_SAMPLES_CV:
        return _fit_platt_with_cv(x_arr, y_arr, n_folds=actual_folds)

    return _fit_platt_single(x_arr, y_arr)


def _fit_platt_single(x_arr: np.ndarray, y_arr: np.ndarray) -> Optional[Tuple[float, float]]:
    """Fit Platt scaling on all data without cross-validation."""
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
    logger.debug("Platt params: a=%.4f b=%.4f (n=%d)", a_val, b_val, x_arr.shape[0])
    return a_val, b_val


def _fit_platt_with_cv(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    n_folds: int = 5,
) -> Optional[Tuple[float, float]]:
    """Fit Platt scaling with k-fold cross-validation, averaging parameters.

    Uses KFold to produce multiple fits and averages the (a, b) parameters.
    Warns if cross-fold variance of parameters is high.
    """
    n = x_arr.shape[0]
    effective_folds = min(n_folds, n)
    if effective_folds < 2:
        return _fit_platt_single(x_arr, y_arr)

    a_vals: list[float] = []
    b_vals: list[float] = []

    try:
        kf = KFold(n_splits=effective_folds, shuffle=True)
        for train_idx, _ in kf.split(x_arr):
            x_train = x_arr[train_idx]
            y_train = y_arr[train_idx]
            if len(set(y_train)) < 2:
                continue
            result = _fit_platt_single(x_train, y_train)
            if result is not None:
                a, b = result
                a_vals.append(a)
                b_vals.append(b)
    except Exception as e:
        logger.warning("Cross-validation Platt fitting failed, falling back to full fit: %s", e)
        return _fit_platt_single(x_arr, y_arr)

    if not a_vals:
        logger.warning("No valid CV folds for Platt scaling, falling back to full fit")
        return _fit_platt_single(x_arr, y_arr)

    a_mean = float(np.mean(a_vals))
    b_mean = float(np.mean(b_vals))
    a_std = float(np.std(a_vals))
    b_std = float(np.std(b_vals))

    cv_warn_threshold = 2.0
    if a_std > cv_warn_threshold or b_std > cv_warn_threshold:
        logger.warning(
            "High Platt parameter variance across %d folds: a=(%.2f +/- %.2f), b=(%.2f +/- %.2f). "
            "Scores may be unstable - consider collecting more calibration data.",
            effective_folds,
            a_mean,
            a_std,
            b_mean,
            b_std,
        )

    logger.debug(
        "Platt params (CV %d-fold): a=%.4f +/- %.4f b=%.4f +/- %.4f (n=%d)",
        effective_folds,
        a_mean,
        a_std,
        b_mean,
        b_std,
        n,
    )
    return a_mean, b_mean


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
