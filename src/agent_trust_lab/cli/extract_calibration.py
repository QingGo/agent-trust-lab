"""Command: extract_calibration"""
import json
from pathlib import Path
from typing import List, Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import setup_logging, cli_verbosity_to_level


@app.command()
def extract_calibration_data(
    result_json: List[str] = typer.Argument(
        ..., help="Path(s) to result JSON file(s) (from --report export). Can specify multiple."
    ),
    output: str = typer.Option(
        "calibration_candidates.json",
        "--output", "-o",
        help="Output JSON path for calibration candidates",
    ),
    target: int = typer.Option(
        200, "--target", "-n", help="Target number of candidates to extract (default: 200)"
    ),
    csv_output: Optional[str] = typer.Option(
        None, "--csv", help="Also export as CSV (for Label Studio)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible sampling"),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Extract calibration annotation candidates from evaluation result JSONs.

    Reads one or more result JSON files (from --report export after run),
    performs stratified sampling to ensure balanced GSAR label coverage,
    and outputs a JSON file that can be loaded by an annotation tool.

    The output JSON contains step_content, evidence, trap_type, and the
    LLM's original classification for each candidate — everything an
    annotator needs to assign ground-truth GSAR labels.

    Examples:
        agent-trust-lab extract-calibration-data results/cmp_3models/flash.json
        agent-trust-lab extract-calibration-data \\
            results/cmp_3models/flash.json results/cmp_3models/pro.json -o my.json
        agent-trust-lab extract-calibration-data results/cmp_3models/flash.json \\
            --csv annotations.csv
    """
    from agent_trust_lab.calibration.extract import (
        build_calibration_candidates_json,
        candidates_to_csv,
    )


    setup_logging(level=cli_verbosity_to_level(verbose))

    if not result_json:
        console.print("[red]Specify at least one result JSON file.[/red]")
        raise typer.Exit(code=1)

    missing = [p for p in result_json if not Path(p).is_file()]
    if missing:
        console.print(f"[red]Files not found: {', '.join(missing)}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Extracting calibration candidates[/bold] from {len(result_json)} file(s)"
    )
    console.print(f"  Target: {target} candidates")
    console.print(f"  Seed: {seed}")

    output_path = build_calibration_candidates_json(
        result_paths=result_json,
        output_path=output,
        target_count=target,
        seed=seed,
    )

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    dist = meta.get("label_distribution", {})
    total = meta.get("total_candidates", 0)

    console.print(f"\n[green]{total} candidates extracted to {output_path}[/green]")
    console.print("  Label distribution:")
    for label in ["Grounded", "Ungrounded", "Contradicted", "Complementary"]:
        count = dist.get(label, 0)
        color = (
            "green" if label == "Grounded"
            else "red" if label in ("Ungrounded", "Contradicted")
            else "yellow"
        )
        console.print(f"    [{color}]{label}[/{color}]: {count}")

    unique_traps = len(set(c.get("trap_id") for c in data.get("candidates", [])))
    unique_types = len(set(c.get("trap_type") for c in data.get("candidates", [])))
    console.print(f"  Unique traps: {unique_traps}")
    console.print(f"  Unique trap types: {unique_types}")

    if csv_output:
        csv_path = candidates_to_csv(output_path, csv_output)
        console.print(f"\n[green]CSV export: {csv_path}[/green]")
