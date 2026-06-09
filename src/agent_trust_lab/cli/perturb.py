"""Command: perturb"""
import json as _json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import cli_verbosity_to_level, get_logger, setup_logging

logger = get_logger("cli.perturb")


@app.command()
def perturb(
    result_json: str = typer.Argument(..., help="Path to result JSON (from --report export)"),
    trap_id: Optional[str] = typer.Option(
        None, "--trap-id", "-t", help="Test a single trap (default: all traps in result)"
    ),
    perturbations: Optional[str] = typer.Option(
        None,
        "--perturbations",
        "-p",
        help="Comma-separated perturbation types: truncate_50,truncate_75,reorder,"
        "remove_middle,noise (default: all)",
    ),
    threshold: float = typer.Option(
        0.15,
        "--threshold",
        help="Stability threshold: score delta above which result is flagged unstable (0-1)",
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for noise perturbation"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Test score stability by perturbing trajectories and re-evaluating.

    Applies controlled perturbations (truncation, reordering, noise insertion)
    to captured trajectories and re-runs the HalluKG pipeline to measure
    whether scores remain stable under input variation.

    A trap is flagged as "unstable" if any perturbation causes a score change
    greater than the stability threshold.

    This is a zero-additional-LLM-cost measurement of GSAR score reliability
    — only deterministic computations (anchoring, NLI, TF-IDF) are re-run.

    Examples:
        agent-trust-lab perturb results/cmp_3models/pro.json
        agent-trust-lab perturb results/cmp_3models/pro.json -t trap_001
        agent-trust-lab perturb results/cmp_3models/pro.json -p truncate_50,reorder
    """

    setup_logging(level=cli_verbosity_to_level(verbose))

    path = Path(result_json)
    if not path.is_file():
        console.print(f"[red]Result file not found: {result_json}[/red]")
        raise typer.Exit(code=1)

    with open(path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No results found in JSON file.[/yellow]")
        return

    if trap_id:
        results = [r for r in results if r.get("trap_id") == trap_id]
        if not results:
            console.print(f"[red]Trap '{trap_id}' not found in results.[/red]")
            raise typer.Exit(code=1)

    pert_names = None
    if perturbations:
        pert_names = [p.strip() for p in perturbations.split(",") if p.strip()]

    console.print(
        f"[bold]Perturbation Stability Test[/bold]"
        f" ({len(results)} trap(s), threshold={threshold})"
    )

    total_unstable = 0
    total_tests = 0

    table = Table(title="Stability Results")
    table.add_column("Trap ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Perturbations", style="dim")
    table.add_column("Max Delta", style="yellow")
    table.add_column("Verdict", style="red")

    for r in results:
        tid = r.get("trap_id", "?")
        tt = r.get("trap_type", "?")
        if not r.get("trajectory_steps"):
            console.print(f"  [yellow]SKIP[/yellow] {tid}: no trajectory_steps in result")
            continue

        try:
            pert_results = _run_perturbation_from_result(
                result=r,
                perturbation_names=pert_names,
                stability_threshold=threshold,
                seed=seed,
            )
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {tid}: {e}")
            continue

        if not pert_results:
            console.print(f"  [dim]SKIP[/dim] {tid}: no perturbations applied")
            continue

        total_tests += len(pert_results)
        max_d = max(pr.get("max_delta", 0) for pr in pert_results)
        unstable = any(pr.get("unstable", False) for pr in pert_results)
        if unstable:
            total_unstable += 1

        pert_str = ", ".join(pr.get("perturbation", "?") for pr in pert_results)
        verdict = "[red]UNSTABLE[/red]" if unstable else "[green]STABLE[/green]"
        table.add_row(tid, tt, pert_str, f"{max_d:.3f}", verdict)

    console.print(table)
    if total_tests > 0:
        pct = 100 * total_unstable / len([r for r in results if r.get("trajectory_steps")])
        console.print(
            f"\nUnstable traps: {total_unstable}/{len(results)} ({pct:.0f}%) — "
            f"higher is worse reliability"
        )


def _run_perturbation_from_result(
    result: dict,
    perturbation_names: Optional[list] = None,
    stability_threshold: float = 0.15,
    seed: Optional[int] = None,
) -> list:
    from agent_trust_lab.hallukg.anchoring import AnchoringReasoner
    from agent_trust_lab.hallukg.faithfulness import FaithfulnessChecker
    from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep
    from agent_trust_lab.perturbation import PERTURBATIONS

    traj_steps = result.get("trajectory_steps", [])
    if not traj_steps:
        return []

    trajectory = SecureTrajectory(
        steps=[TrajectoryStep(type=s.get("type", "unknown"), content=s["content"])
               for s in traj_steps],
        security_events=[],
    )

    names = perturbation_names or list(PERTURBATIONS.keys())
    pert_results = []
    orig_hallu = result.get("hallucination", {})
    orig_scores = {
        "avg_g": orig_hallu.get("avg_g_score", 0),
        "avg_u": orig_hallu.get("avg_u_score", 0),
        "avg_c": orig_hallu.get("avg_c_score", 0),
        "avg_f": orig_hallu.get("avg_faithfulness", 0),
    }

    for name in names:
        perturb_fn = PERTURBATIONS.get(name)
        if perturb_fn is None:
            continue
        try:
            perturbed = perturb_fn(trajectory)
        except Exception as e:
            logger.warning("Perturbation '%s' failed for trap %s: %s", name, trap_id, e)
            continue

        steps = perturbed.steps

        knowledge_source = result.get("metadata", {}).get("knowledge_source", "")
        reasoner = AnchoringReasoner()
        checker = FaithfulnessChecker()

        all_triples = []
        for step in steps:
            anchored = reasoner.batch_anchor(
                [{"subject": "step", "predicate": "says", "object": step.content}],
                knowledge_text=knowledge_source,
            )
            all_triples.extend(anchored)

        pert_g = 0.0
        n_eval = 0
        for i, step in enumerate(steps):
            if step.type in ("harness_init", "trap_injection", "error"):
                continue
            evidence = [t.get("evidence", ["no evidence"]) for t in all_triples]
            flat_evidence = []
            for e in evidence:
                if isinstance(e, list):
                    flat_evidence.extend(e)
                else:
                    flat_evidence.append(str(e))
            nli = checker.check([step.content], flat_evidence or ["no evidence"])
            pert_g += nli
            n_eval += 1

        if n_eval > 0:
            pert_g /= n_eval

        deltas = {
            "g": abs(pert_g - orig_scores["avg_g"]),
        }
        max_delta = deltas["g"]
        unstable = max_delta >= stability_threshold

        pert_results.append({
            "perturbation": name,
            "original_scores": orig_scores,
            "perturbed_g": pert_g,
            "deltas": deltas,
            "max_delta": max_delta,
            "unstable": unstable,
        })

    return pert_results
