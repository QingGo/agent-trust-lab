"""Interactive terminal-based GSAR annotation tool for calibration data."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

GSAR_LABELS = ["Grounded", "Ungrounded", "Contradicted", "Complementary"]
LABEL_SHORTCODES = {"1": "Grounded", "g": "Grounded", "2": "Ungrounded", "u": "Ungrounded",
                     "3": "Contradicted", "c": "Contradicted", "4": "Complementary",
                     "p": "Complementary"}

SCORE_PRESETS: Dict[str, Dict[str, float]] = {
    "Grounded": {
        "g_score": 0.90, "u_score": 0.05, "c_score": 0.05, "faithfulness_score": 0.95
    },
    "Ungrounded": {
        "g_score": 0.10, "u_score": 0.85, "c_score": 0.10, "faithfulness_score": 0.15
    },
    "Contradicted": {
        "g_score": 0.05, "u_score": 0.10, "c_score": 0.90, "faithfulness_score": 0.10
    },
    "Complementary": {
        "g_score": 0.80, "u_score": 0.10, "c_score": 0.05, "faithfulness_score": 0.90
    },
}


def _load_annotations(output_path: str) -> Dict[str, Any]:
    """Load existing annotations, or return empty template."""
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"benchmark": "gsar-calibration-v1", "version": "1.0", "annotations": []}


def _save_annotations(output_path: str, annotations: Dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def _format_score_slider(label: str, value: float, width: int = 20) -> str:
    filled = int(value * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"  {label:>18s}: [{bar}] {value:.2f}"


def _display_candidate(candidate: dict, index: int, total: int) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    trap_id = candidate.get("trap_id", "?")
    step_index = candidate.get("step_index", -1)
    trap_type = candidate.get("trap_type", "?")
    step_type = candidate.get("step_type", "?")
    content = candidate.get("step_content", "")
    evidence = candidate.get("evidence", [])
    explanation = candidate.get("explanation", "")
    original_label = candidate.get("original_gsar_label", "?")
    disagreement = candidate.get("gsar_nli_disagreement", 0.0)

    header = Text()
    header.append(f"[{index + 1}/{total}] ", style="bold cyan")
    header.append(f"{trap_id}", style="cyan")
    header.append(f" step={step_index}  ", style="dim")
    header.append(f"{trap_type}", style="green")
    header.append(f" / {step_type}", style="blue")

    body = Text()
    body.append("\n[bold]Agent Step Content:[/bold]\n")
    body.append(content.strip()[:2000], style="white")
    if len(content) > 2000:
        body.append(f"\n[dim](...truncated, {len(content)} chars total)[/dim]")

    body.append("\n\n[bold]Evidence (Anchored Triples):[/bold]")
    if evidence:
        for e in evidence:
            if isinstance(e, str):
                body.append(f"\n  \u2022 {e[:300]}", style="dim")
            else:
                body.append(f"\n  \u2022 {str(e)[:300]}", style="dim")
    else:
        body.append("\n  (no evidence)", style="dim")

    if explanation:
        body.append("\n\n[bold]LLM Explanation:[/bold]")
        body.append(f"\n  {explanation[:500]}", style="dim")

    footer = Text()
    footer.append(f"\nOriginal GSAR: [bold]{original_label}[/bold]", style="yellow")
    if disagreement >= 0.3:
        footer.append(f"  GSAR-NLI disagreement: [red]{disagreement:.3f}[/red]")
    else:
        footer.append(f"  GSAR-NLI disagreement: [dim]{disagreement:.3f}[/dim]")

    panel = Panel(body, title=header, subtitle=footer, border_style="blue")
    console.print(panel)


def _ask_label() -> Optional[str]:
    """Ask the user for a GSAR label. Returns the label string or None to skip."""
    from rich.console import Console
    from rich.text import Text

    console = Console()
    prompt = Text()
    prompt.append("\nChoose label: ", style="bold")
    prompt.append("[1/G] Grounded  ", style="green")
    prompt.append("[2/U] Ungrounded  ", style="red")
    prompt.append("[3/C] Contradicted  ", style="magenta")
    prompt.append("[4/P] Complementary  ", style="yellow")
    prompt.append("[s] Skip  [q] Quit", style="dim")

    console.print(prompt)
    choice = input("> ").strip().lower()

    if not choice:
        return _ask_label()
    if choice == "s":
        return "SKIP"
    if choice == "q":
        return "QUIT"
    if choice in LABEL_SHORTCODES:
        return LABEL_SHORTCODES[choice]
    console.print("[yellow]Invalid choice. Use 1/G, 2/U, 3/C, 4/P, s, or q.[/yellow]")
    return _ask_label()


def _ask_score(label: str, score_name: str, default: float) -> float:
    """Ask for a confidence score with a preset default."""
    try:
        raw = input(f"  {score_name} [{default:.2f}]> ").strip()
        if not raw:
            return default
        val = float(raw)
        return max(0.0, min(1.0, val))
    except ValueError:
        print(f"  Invalid. Using default: {default:.2f}")
        return default


def _ask_custom_scores(label: str) -> Optional[Dict[str, float]]:
    """Ask user if they want custom scores or use presets."""
    preset = SCORE_PRESETS.get(label, SCORE_PRESETS["Grounded"])
    choice = input("  Use preset scores? [Y/n/custom]> ").strip().lower()
    if choice in ("", "y", "yes"):
        return dict(preset)
    if choice == "n":
        return None
    if choice in ("c", "custom"):
        scores = {}
        scores["g_score"] = _ask_score(label, "g_score", preset["g_score"])
        scores["u_score"] = _ask_score(label, "u_score", preset["u_score"])
        scores["c_score"] = _ask_score(label, "c_score", preset["c_score"])
        scores["faithfulness_score"] = _ask_score(
            label, "faithfulness_score", preset["faithfulness_score"]
        )
        return scores
    return dict(preset)


def run_interactive_annotation(
    candidates_path: str,
    output_path: str,
    auto_save: bool = True,
) -> Dict[str, Any]:
    """Interactive terminal annotation loop.

    Returns:
        Dict with keys: total, annotated, output_path.
    """
    with open(candidates_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data.get("candidates", [])
    total = len(candidates)

    annotations_doc = _load_annotations(output_path)
    existing = set()
    for ann in annotations_doc.get("annotations", []):
        key = (ann.get("trap_id"), ann.get("step_index"))
        existing.add(key)

    annotator = _load_annotations(output_path)
    existing_annotations = annotator.get("annotations", [])
    existing_pairs: Set[Tuple[str, int]] = set()
    for ann in existing_annotations:
        existing_pairs.add((ann.get("trap_id"), ann.get("step_index")))

    start_idx = 0
    for i, candidate in enumerate(candidates):
        key = (candidate.get("trap_id"), candidate.get("step_index"))
        if key not in existing_pairs:
            start_idx = i
            break
    else:
        start_idx = total

    from rich.console import Console

    console = Console()

    if start_idx > 0:
        console.print(
            f"[dim]Resuming: {start_idx}/{total} candidates already annotated.[/dim]\n"
        )

    annotated_count = start_idx

    for i in range(start_idx, total):
        candidate = candidates[i]
        key = (candidate.get("trap_id"), candidate.get("step_index"))

        if key in existing_pairs:
            console.print(f"[dim][{i + 1}/{total}] Already annotated, skipping...[/dim]")
            annotated_count += 1
            continue

        _display_candidate(candidate, i, total)

        label = _ask_label()
        if label == "QUIT":
            if auto_save:
                _save_annotations(output_path, annotator)
                console.print(f"\n[green]Progress saved to {output_path}[/green]")
            break
        if label == "SKIP":
            console.print("[dim]Skipped.[/dim]\n")
            continue
        if label is None:
            continue

        scores = _ask_custom_scores(label)

        score_defaults = dict(scores) if scores else {
            "g_score": 0.0, "u_score": 0.0, "c_score": 0.0, "faithfulness_score": 1.0
        }
        annotation = {
            "trap_id": candidate.get("trap_id"),
            "step_index": candidate.get("step_index"),
            "gsar_label": label,
            "g_score": score_defaults.get("g_score", 0.0),
            "u_score": score_defaults.get("u_score", 0.0),
            "c_score": score_defaults.get("c_score", 0.0),
            "faithfulness_score": score_defaults.get("faithfulness_score", 1.0),
        }
        annotator["annotations"].append(annotation)
        existing_pairs.add(key)
        annotated_count += 1

        label_color = {
            "Grounded": "green", "Ungrounded": "red",
            "Contradicted": "magenta", "Complementary": "yellow",
        }.get(label, "white")
        console.print(f"[{label_color}]  -> {label}[/{label_color}]")
        if scores:
            for sn, sv in scores.items():
                console.print(_format_score_slider(sn, sv))
        console.print()

        if auto_save:
            _save_annotations(output_path, annotator)

    if auto_save:
        _save_annotations(output_path, annotator)

    return {"total": total, "annotated": annotated_count, "output_path": output_path}
