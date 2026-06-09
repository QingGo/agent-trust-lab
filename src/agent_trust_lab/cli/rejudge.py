"""Command: rejudge"""
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import setup_logging, cli_verbosity_to_level


@app.command()
def rejudge(
    result_json: str = typer.Argument(..., help="Path to result JSON (from --report export)"),
    judge: str = typer.Option(
        ..., "--judge", "-j", help="LLM model to use as alternate GSAR judge"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON path (default: rejudged_<input>)"
    ),
    full: bool = typer.Option(
        False, "--full", help="Re-run full HalluKG pipeline (extractor + anchor + GSAR)"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL (default: https://api.deepseek.com)"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Re-evaluate hallucination scores with a different GSAR judge model.

    Reads a result JSON that contains checkpoint data (anchored triples, trajectory
    steps) and re-runs only the GSAR classification step with the specified judge.

    The original harness execution, triple extraction, and anchoring results are
    reused — only the GSAR classifier is re-invoked. This makes cross-judge
    validation extremely cheap (~1 LLM call per trap).

    Use --full to re-run the complete HalluKG pipeline (triple extraction +
    anchoring + GSAR) using the saved trajectory_steps from the result JSON.

    Examples:
        agent-trust-lab rejudge results/cmp_3models/pro.json --judge mimo-v2.5-pro
        agent-trust-lab rejudge results/cmp_3models/pro.json --judge deepseek-v4-pro \\
            --output-dir rejudged_by_pro.json
    """
    import json as _json



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

    orig_config = data.get("config", {})
    orig_judge = orig_config.get("judge_model", "unknown")
    console.print(
        f"[bold]Re-judging with:[/bold] {judge} (original judge: {orig_judge})"
    )
    console.print(f"  Traps to re-evaluate: {len(results)}")

    if full:
        console.print("  [yellow]--full mode: re-running complete HalluKG pipeline[/yellow]")

    rejudged_results = _run_rejudge(
        results=results,
        judge_model=judge,
        base_url=base_url or "",
        full_pipeline=full,
    )

    output_path = output or str(path.with_name(f"rejudged_{path.stem}.json"))

    for r in rejudged_results:
        r.pop("checkpoint", None)
        r.pop("trajectory_steps", None)
        r.pop("security_event_log", None)

    payload = {
        "config": {
            **orig_config,
            "rejudge_judge": judge,
            "original_judge": orig_judge,
            "rejudge_mode": "full" if full else "gsar_only",
        },
        "results": rejudged_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)

    _print_rejudge_comparison(results, rejudged_results, judge, orig_judge)
    console.print(f"\n[green]Re-judged results saved to {output_path}[/green]")


def _run_rejudge(
    results: list,
    judge_model: str,
    base_url: str,
    full_pipeline: bool = False,
) -> list:
    from agent_trust_lab.hallukg.classifier import GSARClassifier
    from agent_trust_lab.llm import get_api_key
    from agent_trust_lab.models.trajectory import TrajectoryStep

    api_key = get_api_key()
    if not api_key:
        console.print("[red]No API key available for GSAR classification.[/red]")
        raise typer.Exit(code=1)

    classifier = GSARClassifier(model=judge_model)
    rejudged = []

    for r in results:
        tid = r.get("trap_id", "?")
        checkpoint = r.get("checkpoint", {})
        hallu = r.get("hallucination", {})
        orig_steps = hallu.get("steps", [])

        anchored_triples = checkpoint.get("anchored_triples", [])
        traj_steps = r.get("trajectory_steps", [])

        if full_pipeline:
            if not traj_steps:
                console.print(f"  [yellow]SKIP[/yellow] {tid}: no trajectory_steps for --full")
                rejudged.append(r)
                continue
            steps = [TrajectoryStep(type=s.get("type", "unknown"), content=s["content"])
                     for s in traj_steps]
            knowledge_source = checkpoint.get("knowledge_source", "")
            new_hallu = _rejudge_full_pipeline(
                steps, knowledge_source, judge_model, api_key, base_url
            )
        elif anchored_triples:
            if not traj_steps or not orig_steps:
                console.print(f"  [yellow]SKIP[/yellow] {tid}: no checkpoint data")
                rejudged.append(r)
                continue
            steps = [TrajectoryStep(type=s.get("type", "unknown"), content=s["content"])
                     for s in traj_steps]
            new_hallu = classifier.classify(steps, anchored_triples)
        else:
            console.print(f"  [yellow]SKIP[/yellow] {tid}: no checkpoint data available")
            rejudged.append(r)
            continue

        rejudged_steps = []
        for h in new_hallu:
            step_dict = {
                "step_index": h.step_index,
                "gsar_label": h.gsar_label,
                "g_score": h.g_score,
                "u_score": h.u_score,
                "c_score": h.c_score,
                "faithfulness_score": h.faithfulness_score,
                "evidence": h.evidence,
                "explanation": h.explanation,
            }
            if h.step_index < len(orig_steps):
                orig = orig_steps[h.step_index]
                step_dict["step_type"] = orig.get("step_type", "unknown")
                step_dict["step_content"] = orig.get("step_content", "")
                step_dict["orig_gsar_label"] = orig.get("gsar_label", "")
                step_dict["orig_g_score"] = orig.get("g_score", 0)
                step_dict["orig_faithfulness_score"] = orig.get("faithfulness_score", 0)
            rejudged_steps.append(step_dict)

        n_steps = len(rejudged_steps)
        avg_g = sum(s["g_score"] for s in rejudged_steps) / n_steps if n_steps else 0
        avg_u = sum(s["u_score"] for s in rejudged_steps) / n_steps if n_steps else 0
        avg_c = sum(s["c_score"] for s in rejudged_steps) / n_steps if n_steps else 0
        avg_f = sum(s["faithfulness_score"] for s in rejudged_steps) / n_steps if n_steps else 0

        rejudged.append({
            **{k: v for k, v in r.items() if k not in ("hallucination", "checkpoint",
                                                         "trajectory_steps", "security_event_log")},
            "hallucination": {
                "step_count": n_steps,
                "avg_g_score": avg_g,
                "avg_u_score": avg_u,
                "avg_c_score": avg_c,
                "avg_faithfulness": avg_f,
                "labels": [s["gsar_label"] for s in rejudged_steps],
                "steps": rejudged_steps,
                "_judge_model": judge_model,
            },
        })
        console.print(f"  {tid}: {n_steps} steps, avg_g={avg_g:.4f}")

    return rejudged


def _rejudge_full_pipeline(
    steps: list,
    knowledge_source: str,
    judge_model: str,
    api_key: str,
    base_url: str,
) -> list:
    from agent_trust_lab.hallukg.anchoring import AnchoringReasoner
    from agent_trust_lab.hallukg.classifier import GSARClassifier
    from agent_trust_lab.hallukg.extractor import TripleExtractor

    extractor = TripleExtractor(model=judge_model)
    reasoner = AnchoringReasoner()
    classifier = GSARClassifier(model=judge_model)

    all_triples = []
    for step in steps:
        triples = extractor.extract(step.content)
        anchored = reasoner.batch_anchor(triples, knowledge_text=knowledge_source)
        all_triples.extend(anchored)

    return classifier.classify(steps, all_triples)


def _print_rejudge_comparison(
    orig_results: list,
    new_results: list,
    judge: str,
    orig_judge: str,
) -> None:

    table = Table(title=f"Re-judge Comparison: {orig_judge} → {judge}")
    table.add_column("Trap ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Old G", style="dim")
    table.add_column("New G", style="magenta")
    table.add_column("Old F", style="dim")
    table.add_column("New F", style="magenta")
    table.add_column("Label Δ", style="yellow")

    total_steps = 0
    label_changes = 0
    for orig, new in zip(orig_results, new_results):
        o_hallu = orig.get("hallucination", {})
        n_hallu = new.get("hallucination", {})
        o_g = o_hallu.get("avg_g_score", 0)
        n_g = n_hallu.get("avg_g_score", 0)
        o_f = o_hallu.get("avg_faithfulness", 0)
        n_f = n_hallu.get("avg_faithfulness", 0)

        o_steps = o_hallu.get("steps", [])
        n_steps = n_hallu.get("steps", [])
        changes = sum(
            1 for os_, ns in zip(o_steps, n_steps)
            if os_.get("gsar_label") != ns.get("gsar_label")
        )
        total_steps += len(n_steps)
        label_changes += changes

        label_str = f"{changes}/{len(n_steps)}" if changes else "-"
        table.add_row(
            new.get("trap_id", "?"),
            new.get("trap_type", "?"),
            f"{o_g:.4f}",
            f"{n_g:.4f}",
            f"{o_f:.4f}",
            f"{n_f:.4f}",
            label_str,
        )

    console.print(table)
    if total_steps > 0:
        console.print(
            f"\nLabel changes: {label_changes}/{total_steps} steps "
            f"({100 * label_changes / total_steps:.1f}%)"
        )
