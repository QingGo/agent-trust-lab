"""Statistical significance and effect size analysis for 76-trap 3-model comparison.

Reads results/cmp_3models/comparison.json and computes:
  1. Per-trap Trust Score paired Wilcoxon + t-test (3 pairwise comparisons)
  2. Cohen's d effect sizes
  3. Per-trap-type Wilcoxon (8 types, n >= 8 each)
  4. Overall ranking with confidence intervals
"""

import json
import statistics
from collections import defaultdict

import numpy as np
from scipy import stats


def load_data(path: str = "results/cmp_3models/comparison.json"):
    with open(path) as f:
        return json.load(f)


def paired_analysis(name_a: str, name_b: str, scores_a: list, scores_b: list) -> dict:
    """Paired Wilcoxon + t-test + Cohen's d for two models."""
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = statistics.mean(diffs)
    median_diff = statistics.median(diffs)
    std_diff = statistics.pstdev(diffs) if n > 1 else 0

    # Wilcoxon signed-rank test
    w_stat, w_p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(scores_a, scores_b)

    # Cohen's d (pooled SD)
    pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # 95% CI for mean difference
    se = std_diff / np.sqrt(n)
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    # Effect size interpretation
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        es_label = "negligible"
    elif d_abs < 0.5:
        es_label = "small"
    elif d_abs < 0.8:
        es_label = "medium"
    else:
        es_label = "large"

    return {
        "name_a": name_a, "name_b": name_b, "n": n,
        "mean_diff": mean_diff, "median_diff": median_diff,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "ttest_stat": t_stat, "ttest_p": t_p,
        "cohens_d": cohens_d, "effect_size": es_label,
        "ci_95": (ci_low, ci_high),
    }


def main():
    data = load_data()

    # Build per-model trust scores per trap
    model_scores: dict[str, list[float]] = defaultdict(list)
    model_violations: dict[str, list[int]] = defaultdict(list)
    model_g: dict[str, list[float]] = defaultdict(list)
    model_u: dict[str, list[float]] = defaultdict(list)

    for r in data["results"]:
        for config_label, s in r["scores"].items():
            g = s["hallucination"].get("avg_g_score", 0)
            u = s["hallucination"].get("avg_u_score", 0)
            c_val = s["hallucination"].get("avg_c_score", 0)
            f = s["hallucination"].get("avg_faithfulness", 0)
            trust = (g + f + (1 - u) + (1 - c_val)) / 4
            model_scores[config_label].append(trust)
            model_g[config_label].append(g)
            model_u[config_label].append(u)
            violation = 1 if s["compliance"].get("overall") != "pass" else 0
            model_violations[config_label].append(violation)

    labels = sorted(model_scores.keys())
    short = {"deepseek-v4-flash (no-think)": "flash",
             "deepseek-v4-pro (no-think)": "pro",
             "mimo-v2.5-pro (no-think)": "mimo"}

    print("=" * 70)
    print("STATISTICAL ANALYSIS: 76 High-Discrimination Traps, 3 Models")
    print("=" * 70)

    # --- 1. Per-model summary ---
    print("\n## 1. Per-Model Trust Score Summary\n")
    print(f"{'Model':<8} {'Mean':>6} {'Median':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
    print("-" * 42)
    for label in labels:
        scores = model_scores[label]
        print(f"{short.get(label, label):<8} {statistics.mean(scores):>6.3f} "
              f"{statistics.median(scores):>6.3f} {statistics.stdev(scores):>6.3f} "
              f"{min(scores):>6.3f} {max(scores):>6.3f}")

    # --- 2. Pairwise Wilcoxon + t-test ---
    print("\n## 2. Pairwise Statistical Tests (Trust Score)\n")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = paired_analysis(
                labels[i], labels[j],
                model_scores[labels[i]], model_scores[labels[j]],
            )
            a_short = short.get(r["name_a"], r["name_a"])
            b_short = short.get(r["name_b"], r["name_b"])
            sig_w = "SIGNIFICANT" if r["wilcoxon_p"] < 0.05 else "not significant"
            sig_t = "SIGNIFICANT" if r["ttest_p"] < 0.05 else "not significant"
            direction = ">" if r["mean_diff"] > 0 else "<"
            print(f"  {a_short} vs {b_short}  (n={r['n']})")
            print(f"    Trust delta:   mean={r['mean_diff']:+.4f}  median={r['median_diff']:+.4f}  95% CI=[{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
            print(f"    Wilcoxon:      W={r['wilcoxon_stat']:.1f}  p={r['wilcoxon_p']:.4f}  [{sig_w}]")
            print(f"    Paired t-test: t={r['ttest_stat']:+.3f}  p={r['ttest_p']:.4f}  [{sig_t}]")
            print(f"    Cohen's d:     {r['cohens_d']:+.3f}  ({r['effect_size']})")
            print(f"    Direction:     {a_short} {direction} {b_short}")
            print()

    # --- 3. Per-component comparison ---
    print("## 3. Per-Component Statistical Tests\n")
    components = {"G Score": model_g, "U Score": model_u}
    for comp_name, comp_data in components.items():
        print(f"  --- {comp_name} ---")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_vals = comp_data[labels[i]]
                b_vals = comp_data[labels[j]]
                w_stat, w_p = stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
                mean_d = statistics.mean([a - b for a, b in zip(a_vals, b_vals)])
                a_s = short[labels[i]]
                b_s = short[labels[j]]
                sig = "**" if w_p < 0.05 else ""
                print(f"    {a_s} vs {b_s}: delta={mean_d:+.4f}  p={w_p:.4f}  {sig}")
        print()

    # --- 4. Per-trap-type analysis ---
    print("## 4. Per-Trap-Type Wilcoxon Tests\n")

    # Get trap_type from scores
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in data["results"]:
        first_scores = list(r["scores"].values())[0]
        tt = first_scores.get("trap_type", "unknown")
        for config_label, s in r["scores"].items():
            g = s["hallucination"].get("avg_g_score", 0)
            u = s["hallucination"].get("avg_u_score", 0)
            c_val = s["hallucination"].get("avg_c_score", 0)
            f = s["hallucination"].get("avg_faithfulness", 0)
            trust = (g + f + (1 - u) + (1 - c_val)) / 4
            by_type[tt][config_label].append(trust)

    print(f"{'Type':<35} {'N':>3} {'flash':>8} {'pro':>8} {'mimo':>8} {'Spread':>8} "
          "{'Best':>10} {'p(flash-pro)':>12} {'p(flash-mimo)':>12} {'p(pro-mimo)':>12}")
    print("-" * 125)

    for tt in sorted(by_type.keys()):
        vals = by_type[tt]
        n = len(list(vals.values())[0])
        means = {}
        for cl in sorted(vals.keys()):
            means[short.get(cl, cl)] = statistics.mean(vals[cl])
        sorted_models = sorted(means.items(), key=lambda x: x[1], reverse=True)
        best = sorted_models[0][0]
        spread = sorted_models[0][1] - sorted_models[-1][1]

        # Pairwise wilcoxon for this type
        w_pairs = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a = vals.get(labels[i], [])
                b = vals.get(labels[j], [])
                if len(a) >= 3:
                    _, w_p = stats.wilcoxon(a, b, alternative="two-sided")
                else:
                    w_p = float("nan")
                key = f"{short[labels[i]]}-{short[labels[j]]}"
                w_pairs[key] = w_p

        f = means.get("flash", 0)
        p = means.get("pro", 0)
        m = means.get("mimo", 0)
        pf = w_pairs.get("flash-pro", float("nan"))
        fm = w_pairs.get("flash-mimo", float("nan"))
        pm = w_pairs.get("pro-mimo", float("nan"))

        print(f"{tt:<35} {n:>3} {f:>8.3f} {p:>8.3f} {m:>8.3f} {spread:>8.3f} "
              f"{best:>10} {pf:>12.4f} {fm:>12.4f} {pm:>12.4f}")

    # --- 5. Power analysis ---
    print("\n## 5. Power Analysis (Paired, alpha=0.05, two-sided)\n")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_scores = model_scores[labels[i]]
            b_scores = model_scores[labels[j]]
            diffs_array = np.array(a_scores) - np.array(b_scores)
            d_obs = np.mean(diffs_array) / np.std(diffs_array, ddof=1) if np.std(diffs_array, ddof=1) > 0 else 0
            # Power for current n
            n = len(a_scores)
            from scipy.stats import nct
            df = n - 1
            t_crit = stats.t.ppf(1 - 0.05 / 2, df)
            nc = d_obs * np.sqrt(n)
            power = 1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc)

            a_s = short[labels[i]]
            b_s = short[labels[j]]
            print(f"  {a_s} vs {b_s}: d={d_obs:+.3f}  n={n}  power={power:.1%}")


if __name__ == "__main__":
    main()
