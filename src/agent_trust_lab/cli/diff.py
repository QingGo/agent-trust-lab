"""Command: diff — compare two evaluation result files."""

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console


def _load_results(path: Path) -> tuple[dict[str, Any], list[dict]]:
    """Load a results JSON file. Returns (metadata, traps_list)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    label = config.get("model", path.stem)
    results = data.get("results", [])
    return {"label": label, "config": config}, results


def _trap_index(results: list[dict]) -> dict[str, dict]:
    """Build a {trap_id: trap_result} index."""
    return {r["trap_id"]: r for r in results}


def _trap_metrics(trap: dict) -> dict[str, float]:
    """Extract per-trap metrics from a result."""
    hallu = trap.get("hallucination", {})
    compliance = trap.get("compliance", {})
    passed = 1.0 if compliance.get("overall") == "pass" else 0.0
    return {
        "pass": passed,
        "G": round(hallu.get("avg_g_score", 0), 3),
        "F": round(hallu.get("avg_faithfulness", 0), 3),
        "U": round(hallu.get("avg_u_score", 0), 3),
        "C": round(hallu.get("avg_c_score", 0), 3),
    }


def _delta_color(delta: float, threshold: float = 0.05) -> Optional[str]:
    """Return a rich color tag for significant deltas."""
    if abs(delta) < threshold:
        return None
    if delta > 0:
        return "green"
    return "red"


@app.command()
def diff(
    file1: Path = typer.Argument(..., help="Path to first results JSON file"),
    file2: Path = typer.Argument(..., help="Path to second results JSON file"),
    threshold: float = typer.Option(
        0.05, "--threshold", "-t", help="Significance threshold for highlighting"
    ),
) -> None:
    """Compare two evaluation result files side by side.

    Highlights traps where metrics differ by more than --threshold (default 0.05).
    Use 'rejudge' to regenerate results with the new scoring before comparing.

    Example:
        agent-trust-lab diff results_old.json results_new.json
        agent-trust-lab diff results_old.json results_new.json --threshold 0.10
    """
    if not file1.exists():
        console.print(f"[red]File not found: {file1}[/red]")
        raise typer.Exit(1)
    if not file2.exists():
        console.print(f"[red]File not found: {file2}[/red]")
        raise typer.Exit(1)

    meta1, traps1 = _load_results(file1)
    meta2, traps2 = _load_results(file2)

    idx1 = _trap_index(traps1)
    idx2 = _trap_index(traps2)

    common = sorted(set(idx1.keys()) & set(idx2.keys()))
    only1 = sorted(set(idx1.keys()) - set(idx2.keys()))
    only2 = sorted(set(idx2.keys()) - set(idx1.keys()))

    if not common:
        console.print("[yellow]No common traps to compare[/yellow]")
        if only1:
            console.print(f"  Only in {file1}: {len(only1)} traps")
        if only2:
            console.print(f"  Only in {file2}: {len(only2)} traps")
        return

    console.print(
        f"\nComparing [bold]{meta1['label']}[/bold] vs [bold]{meta2['label']}[/bold] "
        f"({len(common)} traps in common)\n"
    )

    if only1:
        console.print(f"  [dim]Only in {file1}: {', '.join(only1[:5])}[/dim]")
    if only2:
        console.print(f"  [dim]Only in {file2}: {', '.join(only2[:5])}[/dim]")
    if only1 or only2:
        console.print()

    table = Table(title=f"Per-Trap Delta ({meta2['label']} − {meta1['label']})")
    table.add_column("Trap", style="cyan", no_wrap=True)
    table.add_column("Pass", justify="center")
    table.add_column("ΔG", justify="right")
    table.add_column("ΔF", justify="right")
    table.add_column("ΔU", justify="right")
    table.add_column("ΔC", justify="right")

    agg: dict[str, float] = {"pass": 0, "G": 0, "F": 0, "U": 0, "C": 0}
    sig_count = 0

    for tid in common:
        m1 = _trap_metrics(idx1[tid])
        m2 = _trap_metrics(idx2[tid])
        deltas = {k: round(m2[k] - m1[k], 3) for k in m1}
        for k in agg:
            agg[k] += deltas[k]

        sig = any(abs(d) >= threshold for d in deltas.values())
        if sig:
            sig_count += 1

        pass_str = "✓" if m2["pass"] > m1["pass"] else ("✗" if m2["pass"] < m1["pass"] else "=")
        pass_color = "green" if m2["pass"] > m1["pass"] else ("red" if m2["pass"] < m1["pass"] else "")

        def _fmt(d: float) -> str:
            c = _delta_color(d, threshold)
            sign = "+" if d > 0 else ""
            s = f"{sign}{d:.3f}"
            return f"[{c}]{s}[/{c}]" if c else s

        table.add_row(
            tid,
            f"[{pass_color}]{pass_str}[/{pass_color}]" if pass_color else pass_str,
            _fmt(deltas["G"]),
            _fmt(deltas["F"]),
            _fmt(deltas["U"]),
            _fmt(deltas["C"]),
        )

    console.print(table)
    console.print()

    n = len(common)
    console.print(
        f"  [bold]Average delta[/bold] "
        f"G={agg['G']/n:+.3f}  F={agg['F']/n:+.3f}  "
        f"U={agg['U']/n:+.3f}  C={agg['C']/n:+.3f}  "
        f"Pass={agg['pass']/n:+.3f}"
    )
    console.print(f"  Traps with significant change (>={threshold}): {sig_count}/{n}")
