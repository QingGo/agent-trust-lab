"""IRT (Item Response Theory) trap difficulty and discrimination analysis.

Fits a 2PL IRT model to existing evaluation data to estimate:
- difficulty (b): how hard a trap is for the average model
- discrimination (a): how well a trap separates strong from weak models
- model ability (theta): estimated capability of each evaluated model

Uses maximum likelihood estimation via iterative Newton-Raphson.
No LLM calls required — purely statistical on existing result data.

Usage:
    python scripts/analyze_irt.py results/cmp_3models/comparison.json
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def _irt_probability(theta, a, b):
    return _logistic(a * (theta - b))


def fit_2pl_irt(
    response_matrix: dict[str, dict[str, float]],
    max_iter: int = 200,
    lr: float = 0.1,
    reg: float = 0.01,
) -> dict[str, Any]:
    models = sorted(response_matrix.keys())
    trap_ids = sorted(next(iter(response_matrix.values())).keys())
    n_models = len(models)
    n_traps = len(trap_ids)

    thetas = {m: 0.0 for m in models}
    a_params = {t: 1.0 for t in trap_ids}
    b_params = {t: 0.0 for t in trap_ids}

    for iteration in range(max_iter):
        theta_grad = {m: 0.0 for m in models}
        a_grad = {t: 0.0 for t in trap_ids}
        b_grad = {t: 0.0 for t in trap_ids}

        model_pen = reg * sum(t * t for t in thetas.values())

        for m in models:
            for t in trap_ids:
                y = response_matrix[m][t]
                p = _irt_probability(thetas[m], a_params[t], b_params[t])
                err = y - p
                theta_grad[m] += a_params[t] * err - reg * thetas[m]
                a_grad[t] += (thetas[m] - b_params[t]) * err - reg * a_params[t]
                b_grad[t] += -a_params[t] * err - reg * b_params[t]

        max_grad = max(
            max(abs(g) for g in theta_grad.values()),
            max(abs(g) for g in a_grad.values()),
            max(abs(g) for g in b_grad.values()),
        )
        if max_grad < 1e-6:
            break

        for m in models:
            thetas[m] += lr * theta_grad[m] / n_traps
        for t in trap_ids:
            a_params[t] += lr * a_grad[t] / n_models
            b_params[t] += lr * b_grad[t] / n_models

    return {
        "models": [{"name": m, "theta": round(thetas[m], 4)} for m in models],
        "traps": [
            {
                "trap_id": t,
                "discrimination": round(a_params[t], 4),
                "difficulty": round(b_params[t], 4),
                "info_at_theta_0": round(
                    a_params[t] ** 2 * _irt_probability(0, a_params[t], b_params[t]) *
                    (1 - _irt_probability(0, a_params[t], b_params[t])), 4
                ),
            }
            for t in trap_ids
        ],
        "iterations": iteration + 1,
    }


def load_response_matrix(comparison_path: str) -> dict[str, dict[str, float]]:
    with open(comparison_path) as f:
        data = json.load(f)

    response_matrix: dict[str, dict[str, float]] = {}
    for r in data.get("results", []):
        tid = r.get("trap_id", "")
        scores = r.get("scores", {})
        for label, s in scores.items():
            h = s.get("hallucination", {})
            g = h.get("avg_g_score", 0)
            u = h.get("avg_u_score", 0)
            c_val = h.get("avg_c_score", 0)
            f_val = h.get("avg_faithfulness", 0)
            trust = (g + f_val + (1 - u) + (1 - c_val)) / 4
            clean_label = label.replace(" (no-think)", "").replace(" (think ", " (").strip()
            if "mimo" in clean_label.lower():
                clean_label = "mimo-v2.5-pro"
            elif "pro" in clean_label.lower():
                clean_label = "deepseek-v4-pro"
            elif "flash" in clean_label.lower():
                clean_label = "deepseek-v4-flash"
            if clean_label not in response_matrix:
                response_matrix[clean_label] = {}
            response_matrix[clean_label][tid] = trust

    return response_matrix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IRT trap difficulty analysis")
    parser.add_argument("comparison_path", help="Path to comparison.json")
    parser.add_argument("--output", default="", help="Output JSON path")
    args = parser.parse_args()

    rm = load_response_matrix(args.comparison_path)
    n_models = len(rm)
    n_traps = len(next(iter(rm.values())))
    print(f"Response matrix: {n_models} models x {n_traps} traps")

    result = fit_2pl_irt(rm)

    models = result["models"]
    traps = result["traps"]

    print(f"\n === Model Ability (theta) ===")
    for m in sorted(models, key=lambda x: x["theta"], reverse=True):
        print(f"  {m['name']}: theta={m['theta']:.4f}")

    print(f"\n === Trap Discriminability ===")
    top_a = sorted(traps, key=lambda x: x["discrimination"], reverse=True)[:5]
    bottom_a = sorted(traps, key=lambda x: x["discrimination"])[:5]
    print("  Top discriminators (high a):")
    for t in top_a:
        print(f"    {t['trap_id']}: a={t['discrimination']:.4f}, b={t['difficulty']:.4f}")
    print("  Bottom discriminators (low a):")
    for t in bottom_a:
        print(f"    {t['trap_id']}: a={t['discrimination']:.4f}, b={t['difficulty']:.4f}")

    print(f"\n === Trap Difficulty ===")
    hardest = sorted(traps, key=lambda x: x["difficulty"], reverse=True)[:3]
    easiest = sorted(traps, key=lambda x: x["difficulty"])[:3]
    print("  Hardest (high b):")
    for t in hardest:
        print(f"    {t['trap_id']}: b={t['difficulty']:.4f}")
    print("  Easiest (low b):")
    for t in easiest:
        print(f"    {t['trap_id']}: b={t['difficulty']:.4f}")

    low_disc = [t for t in traps if t["discrimination"] < 0.1]
    if low_disc:
        print(f"\n === Low-Discrimination Traps ({len(low_disc)}) ===")
        print("  These traps should be reviewed:")
        for t in low_disc:
            print(f"    {t['trap_id']}: a={t['discrimination']:.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nIRT results saved to {args.output}")


if __name__ == "__main__":
    main()
