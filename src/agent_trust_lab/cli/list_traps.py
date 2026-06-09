"""Command: list_traps"""
from typing import Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console
from agent_trust_lab.cli._shared import _get_trap_manager


@app.command()
def list_traps(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category (general_agent, code_agent)"
    ),
    difficulty: Optional[str] = typer.Option(
        None, "--difficulty", "-d", help="Filter by difficulty (trivial, easy, medium, hard)"
    ),
    trap_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by trap type"),
    include_controls: bool = typer.Option(
        False, "--include-controls", help="Include benign control and overly cautious samples"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List available traps with optional filtering."""
    mgr = _get_trap_manager()
    traps = mgr.load_traps(
        category=category,
        difficulty=difficulty,
        include_controls=include_controls,
    )

    if trap_type:
        traps = [t for t in traps if t.trap_type == trap_type]

    if json_output:
        import json

        data = [
            {
                "trap_id": t.trap_id,
                "trap_type": t.trap_type,
                "severity": t.severity,
                "difficulty": t.difficulty,
                "category": t.category,
                "base_task": t.base_task,
            }
            for t in traps
        ]
        console.print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if not traps:
        console.print("[yellow]No traps found matching the criteria.[/yellow]")
        return

    table = Table(title=f"Traps ({len(traps)} found)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Severity", style="red")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Category", style="blue")
    table.add_column("Task Preview", style="dim", max_width=60)

    for t in traps:
        preview = t.base_task[:57] + "..." if len(t.base_task) > 60 else t.base_task
        table.add_row(
            t.trap_id,
            t.trap_type,
            t.severity,
            t.difficulty,
            t.category,
            preview,
        )

    console.print(table)
