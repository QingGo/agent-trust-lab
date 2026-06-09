"""Command: annotate"""
from pathlib import Path

import typer

from agent_trust_lab.cli import app, console


@app.command()
def annotate(
    candidates_json: str = typer.Argument(
        ..., help="Path to calibration candidates JSON (from extract-calibration-data)"
    ),
    output: str = typer.Option(
        "annotations.json",
        "--output", "-o",
        help="Output path for annotations JSON",
    ),
    auto_save: bool = typer.Option(
        True, "--auto-save/--no-auto-save", help="Auto-save progress after each annotation"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Interactively annotate calibration candidates for GSAR labels.

    Walks through each candidate step-by-step in the terminal, showing the
    agent's step content, available evidence, and anchor context. The user
    assigns a GSAR label (Grounded/Ungrounded/Contradicted/Complementary)
    with optional confidence scores.

    Annotations are saved incrementally with --auto-save (default on).
    You can interrupt at any time with Ctrl+C and resume later — progress
    is tracked in the output file.

    Controls:
      1/G = Grounded      2/U = Ungrounded
      3/C = Contradicted  4/P = Complementary
      s = Skip this item   q = Quit and save

    Examples:
        agent-trust-lab annotate calibration_candidates.json
        agent-trust-lab annotate calibration_candidates.json -o my_annotations.json
    """
    if not Path(candidates_json).is_file():
        console.print(f"[red]Candidates file not found: {candidates_json}[/red]")
        raise typer.Exit(code=1)

    from agent_trust_lab.calibration.annotator import run_interactive_annotation

    console.print("[bold]GSAR Calibration Annotation Tool[/bold]")
    console.print(f"  Source: {candidates_json}")
    console.print(f"  Output: {output}")
    console.print(f"  Auto-save: {'on' if auto_save else 'off'}")
    console.print()
    console.print("[dim]Controls: 1=Grounded 2=Ungrounded 3=Contradicted 4=Complementary[/dim]")
    console.print("[dim]          s=Skip  q=Quit and save[/dim]")
    console.print()

    try:
        result = run_interactive_annotation(
            candidates_path=candidates_json,
            output_path=output,
            auto_save=auto_save,
        )
    except (SystemExit, KeyboardInterrupt):
        console.print("\n[yellow]Annotation interrupted. Progress saved.[/yellow]")
        return

    total = result.get("total", 0)
    annotated_count = result.get("annotated", 0)
    console.print(
        f"\n[green]Done: {annotated_count}/{total} candidates annotated.[/green]"
    )
    console.print(f"  Saved to: {output}")
