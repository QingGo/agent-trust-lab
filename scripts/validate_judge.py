"""Validate GSAR classifier reliability against a golden test set.

Usage:
    .venv/bin/python scripts/validate_judge.py [--model MODEL] [--golden PATH]
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_golden(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Golden data must be a JSON array")
    return data


def classify_sample(
    classifier: Any, step_content: str
) -> str:
    from agent_trust_lab.models.trajectory import TrajectoryStep

    step = TrajectoryStep(type="test", content=step_content)
    dummy_triples: List[Dict[str, Any]] = []
    results = classifier.classify([step], dummy_triples)
    if not results:
        return "Grounded"
    return results[0].gsar_label


def compute_cohens_kappa(confusion: Dict[str, Dict[str, int]]) -> float:
    labels = sorted(set(confusion.keys()) | {
        k for v in confusion.values() for k in v
    })
    total = sum(sum(v.values()) for v in confusion.values())
    if total == 0:
        return 0.0

    p_o = sum(confusion.get(l, {}).get(l, 0) for l in labels) / total

    p_e = 0.0
    for l in labels:
        row = sum(confusion.get(l, {}).values())
        col = sum(confusion.get(l2, {}).get(l, 0) for l2 in labels)
        p_e += (row * col) / (total * total)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def validate(
    golden_path: str,
    model: str = "deepseek-v4-flash",
) -> Dict[str, Any]:
    from agent_trust_lab.hallukg.classifier import GSARClassifier
    from agent_trust_lab.llm import get_api_key, reset_token_usage

    api_key = get_api_key()
    if not api_key:
        print("Warning: No API key found. Results will use stub fallback.", file=sys.stderr)

    reset_token_usage()

    golden = load_golden(golden_path)
    classifier = GSARClassifier(model=model)

    confusion: Dict[str, Dict[str, int]] = {}
    correct = 0
    mismatches: List[Dict[str, Any]] = []

    for i, sample in enumerate(golden):
        content = sample["step_content"]
        expected = sample["expected_label"]

        try:
            predicted = classify_sample(classifier, content)
        except Exception as e:
            predicted = "ERROR"
            print(f"  Sample {i}: classification failed: {e}", file=sys.stderr)

        if predicted not in confusion:
            confusion[predicted] = {}
        if expected not in confusion[predicted]:
            confusion[predicted][expected] = 0
        confusion[predicted][expected] += 1

        if predicted == expected:
            correct += 1
        else:
            mismatches.append({
                "index": i,
                "expected": expected,
                "predicted": predicted,
                "context": sample.get("context", ""),
                "content_preview": content[:100],
            })

    kappa = compute_cohens_kappa(confusion)
    accuracy = correct / len(golden) if golden else 0.0

    from agent_trust_lab.llm import get_token_usage

    token_usage = get_token_usage()
    total_tokens = sum(
        details.get("prompt_tokens", 0) + details.get("completion_tokens", 0)
        for details in token_usage.values()
    )

    return {
        "model": model,
        "total_samples": len(golden),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": {
            pred: {exp: confusion[pred].get(exp, 0) for exp in sorted(
                {k for v in confusion.values() for k in v}
            )}
            for pred in sorted(confusion.keys())
        },
        "mismatches": mismatches[:10],
        "token_usage": {
            "total_tokens": total_tokens,
            "per_model": token_usage,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate GSAR classifier reliability against golden labels"
    )
    parser.add_argument(
        "--model", default="deepseek-v4-flash", help="Model for GSAR classification"
    )
    parser.add_argument(
        "--golden",
        default="tests/data/gsar_golden.json",
        help="Path to golden test data JSON",
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output JSON path for results"
    )
    args = parser.parse_args()

    golden_path = args.golden
    if not Path(golden_path).is_file():
        golden_path = Path(__file__).parent.parent / args.golden
        if not golden_path.is_file():
            print(f"Golden data not found: {args.golden}", file=sys.stderr)
            sys.exit(1)

    print(f"Validating GSAR classifier (model={args.model})...")
    print(f"Golden data: {golden_path}")

    result = validate(str(golden_path), args.model)

    print(f"\nSamples: {result['total_samples']}")
    print(f"Correct: {result['correct']} / {result['total_samples']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Cohen's Kappa: {result['cohens_kappa']:.4f}")
    print(f"Tokens used: {result['token_usage']['total_tokens']}")

    if result["cohens_kappa"] >= 0.8:
        print("Verdict: STRONG agreement (kappa >= 0.8)")
    elif result["cohens_kappa"] >= 0.6:
        print("Verdict: MODERATE agreement (0.6 <= kappa < 0.8) — investigate")
    else:
        print("Verdict: WEAK agreement (kappa < 0.6) — judge may be unreliable")

    if result["mismatches"]:
        print(f"\nMismatches ({len(result['mismatches'])}):")
        for m in result["mismatches"][:5]:
            print(
                f"  [{m['index']}] Expected={m['expected']} Got={m['predicted']} "
                f"({m['context']})"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    sys.exit(0 if result["cohens_kappa"] >= 0.6 else 1)


if __name__ == "__main__":
    main()
