"""Calibration profile storage and loading."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent_trust_lab.log import get_logger

logger = get_logger("calibration.profile")

DEFAULT_CALIBRATION_DIR = os.path.expanduser("~/.cache/agent-trust-lab/calibration")


@dataclass
class CalibrationProfile:
    """Stores Platt scaling parameters and Cohen's kappa values for score calibration."""

    profile_id: str
    benchmark: str
    version: str
    created_at: str = ""
    sample_count: int = 0
    kappa_gsar: float = 0.0
    kappa_gsar_ci: Tuple[float, float] = (0.0, 0.0)
    kappa_compliance: float = 0.0
    platt_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_calibrated_score(self, score_name: str, raw_score: float) -> Optional[float]:
        """Apply Platt scaling to a raw score.

        Args:
            score_name: One of g_score, u_score, c_score, faithfulness_score.
            raw_score: Raw model score in [0, 1].

        Returns:
            Calibrated score in [0, 1], or None if no params for this score.
        """
        from agent_trust_lab.calibration.scaler import apply_calibrated_score

        params = self.platt_params.get(score_name)
        if params is None:
            return None
        a = params.get("A", 0.0)
        b = params.get("B", 0.0)
        return round(apply_calibrated_score(raw_score, a, b), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "benchmark": self.benchmark,
            "version": self.version,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "kappa_gsar": self.kappa_gsar,
            "kappa_gsar_ci": list(self.kappa_gsar_ci),
            "kappa_compliance": self.kappa_compliance,
            "platt_params": self.platt_params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProfile":
        ci = data.get("kappa_gsar_ci", [0.0, 0.0])
        return cls(
            profile_id=data.get("profile_id", ""),
            benchmark=data.get("benchmark", ""),
            version=data.get("version", ""),
            created_at=data.get("created_at", ""),
            sample_count=data.get("sample_count", 0),
            kappa_gsar=data.get("kappa_gsar", 0.0),
            kappa_gsar_ci=(float(ci[0]), float(ci[1])),
            kappa_compliance=data.get("kappa_compliance", 0.0),
            platt_params=data.get("platt_params", {}),
            metadata=data.get("metadata", {}),
        )


def _ensure_dir() -> str:
    os.makedirs(DEFAULT_CALIBRATION_DIR, exist_ok=True)
    return DEFAULT_CALIBRATION_DIR


def save_profile(profile: CalibrationProfile) -> str:
    """Save a calibration profile to disk.

    Returns the file path where the profile was saved.
    """
    dir_path = _ensure_dir()
    file_path = os.path.join(dir_path, f"{profile.profile_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Calibration profile saved to %s", file_path)
    return file_path


def load_profile(profile_id: str) -> Optional[CalibrationProfile]:
    """Load a calibration profile by ID.

    Searches ~/.cache/agent-trust-lab/calibration/<profile_id>.json.
    """
    file_path = os.path.join(DEFAULT_CALIBRATION_DIR, f"{profile_id}.json")
    if not os.path.isfile(file_path):
        logger.warning("Calibration profile not found: %s", file_path)
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = CalibrationProfile.from_dict(data)
        logger.debug("Loaded calibration profile %s (n=%d)", profile_id, profile.sample_count)
        return profile
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("Failed to load calibration profile %s: %s", profile_id, e)
        return None


def list_profiles() -> List[str]:
    """List available calibration profile IDs."""
    dir_path = DEFAULT_CALIBRATION_DIR
    if not os.path.isdir(dir_path):
        return []
    profiles = []
    for fname in os.listdir(dir_path):
        if fname.endswith(".json"):
            profiles.append(fname[:-5])
    return sorted(profiles)


def compute_cohens_kappa(
    labels_a: List[str],
    labels_b: List[str],
) -> Tuple[float, Tuple[float, float]]:
    """Compute Cohen's kappa for agreement between two sets of categorical labels.

    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.

    Args:
        labels_a: First rater's labels (e.g., model GSAR labels).
        labels_b: Second rater's labels (e.g., human GSAR labels).

    Returns:
        (kappa, (ci_lower, ci_upper)) tuple.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"Length mismatch: {len(labels_a)} vs {len(labels_b)}")
    if len(labels_a) < 2:
        return 0.0, (0.0, 0.0)

    categories = sorted(set(labels_a) | set(labels_b))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    matrix = [[0] * n_cats for _ in range(n_cats)]
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    total = len(labels_a)
    p_o = sum(matrix[i][i] for i in range(n_cats)) / total

    row_sums = [sum(matrix[i]) for i in range(n_cats)]
    col_sums = [sum(matrix[j][i] for j in range(n_cats)) for i in range(n_cats)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(n_cats)) / (total * total)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0, (0.0, 0.0)

    kappa = (p_o - p_e) / (1.0 - p_e)

    se = _kappa_se(p_o, p_e, total)
    ci_lower = max(-1.0, kappa - 1.96 * se)
    ci_upper = min(1.0, kappa + 1.96 * se)

    return round(kappa, 4), (round(ci_lower, 4), round(ci_upper, 4))


def _kappa_se(p_o: float, p_e: float, n: int) -> float:
    """Standard error of Cohen's kappa."""
    if n <= 1:
        return 1.0
    variance = (p_o * (1.0 - p_o)) / (n * (1.0 - p_e) * (1.0 - p_e))
    if variance < 0:
        return 1.0
    return variance**0.5


def _load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_annotations_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_calibration(
    results_json_path: str,
    annotations_json_path: str,
    profile_id: str = "default",
) -> CalibrationProfile:
    """Run calibration: fit Platt scaling + compute Cohen's kappa from results + human annotations.

    Args:
        results_json_path: Path to JSON export from orchestrator (--report output).
        annotations_json_path: Path to human annotations JSON.
        profile_id: Identifier for the saved calibration profile.

    Returns:
        CalibrationProfile with fitted parameters.

    Raises:
        ValueError: If no matching step pairs found.
    """
    results_data = _load_results_json(results_json_path)
    annotations_data = _load_annotations_json(annotations_json_path)

    raw_results = results_data.get("results", [])
    annotations = annotations_data.get("annotations", [])

    pairs = _align_results_with_annotations(raw_results, annotations)
    if not pairs:
        raise ValueError(
            "No matching (trap_id, step_index) pairs found between results and annotations"
        )

    model_gsar_labels: List[str] = []
    human_gsar_labels: List[str] = []
    score_names = ["g_score", "u_score", "c_score", "faithfulness_score"]
    raw_score_buckets: Dict[str, List[float]] = {s: [] for s in score_names}

    for pair in pairs:
        raw = pair["raw"]
        ann = pair["annotation"]

        raw_gsar = raw.get("gsar_label", "")
        human_gsar = ann.get("gsar_label", "")
        model_gsar_labels.append(raw_gsar)
        human_gsar_labels.append(human_gsar)

        for key in score_names:
            raw_score_buckets[key].append(raw.get(key, 0.0))

    kappa_gsar, kappa_ci = compute_cohens_kappa(model_gsar_labels, human_gsar_labels)

    from agent_trust_lab.calibration.scaler import fit_platt_scaling

    platt_params: Dict[str, Dict[str, float]] = {}
    for score_name in score_names:
        raw_scores = raw_score_buckets[score_name]
        binary_labels = [
            1 if ann.get(score_name, 0.0) >= 0.5 else 0 for ann in (p["annotation"] for p in pairs)
        ]
        result = fit_platt_scaling(raw_scores, binary_labels)
        if result:
            a, b = result
            platt_params[score_name] = {"A": round(a, 6), "B": round(b, 6)}

    benchmark = annotations_data.get("benchmark", "unknown")
    version = annotations_data.get("version", "0.0")

    profile = CalibrationProfile(
        profile_id=profile_id,
        benchmark=benchmark,
        version=version,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC"),
        sample_count=len(pairs),
        kappa_gsar=kappa_gsar,
        kappa_gsar_ci=kappa_ci,
        kappa_compliance=0.0,
        platt_params=platt_params,
    )

    save_profile(profile)
    return profile


def _apply_calibration_to_results(
    data: Dict[str, Any],
    profile: CalibrationProfile,
) -> Dict[str, Any]:
    """Apply calibrated scores to a results JSON payload.

    Adds calibrated_* fields alongside raw scores in each step.
    """
    import copy

    calibrated_data = copy.deepcopy(data)
    for result in calibrated_data.get("results", []):
        hallu = result.get("hallucination")
        if not hallu:
            continue
        for step in hallu.get("steps", []):
            for score_name in ("g_score", "u_score", "c_score", "faithfulness_score"):
                raw = step.get(score_name, 0.0)
                cal = profile.get_calibrated_score(score_name, raw)
                if cal is not None:
                    step[f"calibrated_{score_name}"] = cal
    calibrated_data["calibration"] = profile.to_dict()
    return calibrated_data


def _align_results_with_annotations(
    raw_results: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Match result steps with human annotations by (trap_id, step_index)."""
    result_steps: Dict[tuple, Dict[str, Any]] = {}

    for r in raw_results:
        trap_id = r.get("trap_id", "")
        hallu = r.get("hallucination")
        if not hallu:
            continue
        for step in hallu.get("steps", []):
            key = (trap_id, step.get("step_index", -1))
            result_steps[key] = step

    pairs = []
    for ann in annotations:
        key = (ann.get("trap_id", ""), ann.get("step_index", -1))
        raw_step = result_steps.get(key)
        if raw_step is not None:
            pairs.append({"raw": raw_step, "annotation": ann})

    return pairs
