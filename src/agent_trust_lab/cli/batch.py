"""Command: batch"""
from pathlib import Path
from typing import Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
from agent_trust_lab.cli._shared import _get_traps_data_dir


@app.command()
def batch(
    config_path: str = typer.Argument(..., help="Path to batch YAML configuration file"),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
    ),
):
    """Run batch evaluation with multiple agent configs from a YAML file.

    Executes each evaluation specification in sequence, exports individual
    JSON results, auto-merges them, and generates a comparison report.

    Example batch.yaml:

    \b
    evaluations:
      - label: "flash-baseline"
        model: "deepseek-v4-flash"
        traps: {category: "general_agent", limit: 5}
      - label: "flash-thinking"
        model: "deepseek-v4-flash"
        thinking_enabled: true
        reasoning_effort: "high"
        traps: {category: "general_agent", limit: 5}
    report:
      format: html
      lang: en
    """
    from agent_trust_lab.batch import parse_batch_yaml, run_batch


    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    if not Path(config_path).is_file():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(code=1)

    try:
        batch_config = parse_batch_yaml(config_path)
    except ValueError as e:
        console.print(f"[red]Invalid batch config: {e}[/red]")
        raise typer.Exit(code=1)

    if not batch_config.trap_library_path:
        batch_config.trap_library_path = str(_get_traps_data_dir())

    if batch_config.report_lang == "both":
        pass
    elif batch_config.report_lang.startswith("zh"):
        batch_config.report_lang = "zh"

    fmt = batch_config.report_format
    if fmt not in ("html", "markdown", "md"):
        console.print(f"[red]Invalid report format: {fmt}. Use 'html' or 'markdown'.[/red]")
        raise typer.Exit(code=1)

    ev_labels = ", ".join(s.label for s in batch_config.evaluations)
    console.print(f"[bold]Batch evaluation:[/bold] {ev_labels}")
    console.print(
        f"  {len(batch_config.evaluations)} config(s), output dir: {batch_config.output_dir}"
    )
    console.print(f"  Report: {batch_config.report_format} ({batch_config.report_lang})")

    merged = run_batch(batch_config)

    n_traps = len(merged.get("results", []))
    n_models = len(merged.get("configs", []))
    console.print(
        f"\n[green]Batch complete.[/green] "
        f"{n_traps} traps x {n_models} models."
        f"\n  Output: {batch_config.output_dir}"
    )
