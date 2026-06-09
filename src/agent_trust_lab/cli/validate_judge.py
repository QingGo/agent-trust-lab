"""Command: validate_judge"""
import json
from pathlib import Path
from typing import Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.config import DEFAULT_MODEL


@app.command()
def validate_judge(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="Model for GSAR classification"
    ),
    golden: str = typer.Option(
        "tests/data/gsar_golden.json",
        "--golden",
        "-g",
        help="Path to golden test data JSON",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON path for detailed results"
    ),
):
    """Validate GSAR classifier reliability against a golden test set.

    Evaluates the judge's own classification accuracy using manually labeled
    samples. Reports Cohen's Kappa for inter-rater agreement strength.
    """
    golden_path = golden
    if not Path(golden_path).is_file():
        import agent_trust_lab

        golden_path = str(Path(agent_trust_lab.__file__).parent.parent / golden)
        if not Path(golden_path).is_file():
            console.print(f"[red]Golden data not found: {golden}[/red]")
            raise typer.Exit(code=1)

    console.print(
        f"[bold]Validating GSAR classifier[/bold] (model={model}, {golden_path})"
    )

    from agent_trust_lab.hallukg.classifier import GSARClassifier
    from agent_trust_lab.llm import get_api_key, get_token_usage, reset_token_usage
    from agent_trust_lab.models.trajectory import TrajectoryStep

    api_key = get_api_key()
    if not api_key:
        console.print("[yellow]Warning: No API key found. Using stub fallback.[/yellow]")

    reset_token_usage()

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    classifier = GSARClassifier(model=model)
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    mismatches: list[dict[str, object]] = []

    for i, sample in enumerate(golden):
        sample_dict: dict[str, object] = sample if isinstance(sample, dict) else {}
        content = str(sample_dict.get("step_content", ""))
        expected = str(sample_dict.get("expected_label", ""))
        step = TrajectoryStep(type="test", content=content)
        results = classifier.classify([step], [])
        predicted = results[0].gsar_label if results else "Grounded"

        if predicted not in confusion:
            confusion[predicted] = {}
        confusion[predicted][expected] = confusion[predicted].get(expected, 0) + 1

        if predicted == expected:
            correct += 1
        else:
            mismatches.append({
                "index": i,
                "expected": expected,
                "predicted": predicted,
                "context": str(sample_dict.get("context", "")),
                "content_preview": content[:100],
            })

    labels = sorted({lb for c in confusion for lb in confusion.get(c, {})} | set(confusion.keys()))
    total = sum(sum(v.values()) for v in confusion.values())
    p_o = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in labels) / total if total else 0.0
    p_e = sum(
        sum(confusion.get(lbl, {}).values())
        * sum(confusion.get(lbl2, {}).get(lbl, 0) for lbl2 in labels)
        / (total * total)
        for lbl in labels
    ) if total else 0.0
    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
    accuracy = correct / len(golden) if golden else 0.0

    token_usage = get_token_usage()
    total_tokens = sum(
        d.get("prompt_tokens", 0) + d.get("completion_tokens", 0)
        for d in token_usage.values()
    )

    console.print(f"\nSamples: {len(golden)}")
    console.print(f"Correct: {correct} / {len(golden)}")
    console.print(f"Accuracy: [bold]{accuracy:.1%}[/bold]")
    console.print(f"Cohen's Kappa: [bold]{kappa:.4f}[/bold]")
    console.print(f"Tokens used: {total_tokens}")

    if kappa >= 0.8:
        console.print("[green]Verdict: STRONG agreement (kappa >= 0.8)[/green]")
    elif kappa >= 0.6:
        console.print("[yellow]Verdict: MODERATE agreement (0.6 <= kappa < 0.8)[/yellow]")
    else:
        console.print("[red]Verdict: WEAK agreement (kappa < 0.6)[/red]")

    if mismatches:
        console.print(f"\n[yellow]Mismatches ({len(mismatches)}):[/yellow]")
        for m in mismatches[:5]:
            console.print(
                f"  [{m['index']}] Expected={m['expected']} Got={m['predicted']} "
                f"({m['context']})"
            )

    if output:
        result = {
            "model": model,
            "total_samples": len(golden),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "cohens_kappa": round(kappa, 4),
            "confusion_matrix": confusion,
            "mismatches": mismatches[:10],
            "token_usage": {"total_tokens": total_tokens, "per_model": token_usage},
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Results saved to {output}[/green]")

    if kappa < 0.6:
        raise typer.Exit(code=1)
