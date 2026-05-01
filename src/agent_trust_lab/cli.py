"""CLI entry point for agent-trust-lab."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    name="agent-trust-lab",
    help="Agent reliability and hallucination evaluation toolkit.",
    no_args_is_help=True,
)

console = Console()


def _get_traps_data_dir() -> Path:
    """Resolve the path to the traps data directory."""
    import agent_trust_lab.traps

    pkg_dir = Path(agent_trust_lab.traps.__file__).parent
    data_dir = pkg_dir / "data"
    if data_dir.is_dir():
        return data_dir

    import importlib.resources

    return importlib.resources.files("agent_trust_lab.traps") / "data"


def _get_trap_manager():
    """Lazy-load the trap manager."""
    from agent_trust_lab.traps.manager import TrapManager

    return TrapManager(str(_get_traps_data_dir()))


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


@app.command()
def show_trap(
    trap_id: str = typer.Argument(..., help="The trap ID to display"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw YAML content"),
):
    """Display full details of a single trap."""
    mgr = _get_trap_manager()
    trap = mgr.get_trap(trap_id)

    if trap is None:
        console.print(f"[red]Trap '{trap_id}' not found.[/red]")
        raise typer.Exit(code=1)

    if raw:
        trap_path = _get_traps_data_dir()
        for root, _dirs, files in os.walk(trap_path):
            candidate = f"{trap_id}.yaml"
            if candidate in files:
                filepath = os.path.join(root, candidate)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
                console.print(syntax)
                return

        # Fallback: search more broadly
        console.print("[yellow]Raw YAML file not found by exact match.[/yellow]")
        traps = mgr.load_traps(trap_ids=[trap_id])
        if traps:
            import json

            console.print(json.dumps(traps[0].model_dump(), indent=2, ensure_ascii=False))
        return

    # Rich display
    console.print(f"\n[bold cyan]Trap: {trap.trap_id}[/bold cyan]")
    console.print(f"  [dim]Version:[/dim] {trap.version}")
    console.print(f"  [dim]Type:[/dim] {trap.trap_type}")
    severity_color = (
        "red"
        if trap.severity == "high"
        else "yellow"
        if trap.severity == "medium"
        else "green"
    )
    console.print(f"  [dim]Severity:[/dim] [{severity_color}]{trap.severity}[/]")
    console.print(f"  [dim]Difficulty:[/dim] {trap.difficulty}")
    console.print(f"  [dim]Category:[/dim] {trap.category}")

    console.print("\n[bold]Base Task:[/bold]")
    console.print(f"  {trap.base_task}")

    if trap.tools:
        console.print("\n[bold]Tools:[/bold]")
        for tool in trap.tools:
            console.print(f"  - {tool.get('name', str(tool))}")

    if trap.trap_injection:
        console.print("\n[bold]Trap Injection:[/bold]")
        console.print(f"  {trap.trap_injection}")

    if trap.expected_violation_signature:
        console.print("\n[bold]Expected Violation:[/bold]")
        console.print(f"  {trap.expected_violation_signature}")

    if trap.judgement_hint:
        console.print("\n[bold]Judgement Hint:[/bold]")
        console.print(f"  {trap.judgement_hint}")

    if trap.remediation:
        console.print("\n[bold green]Remediation:[/bold green]")
        console.print(f"  [bold]Problem:[/bold] {trap.remediation.problem}")
        console.print(f"  [bold]Cause:[/bold] {trap.remediation.cause}")
        console.print("  [bold]Fix:[/bold]")
        for line in trap.remediation.fix.strip().split("\n"):
            console.print(f"    {line.strip()}")

    if trap.variation_rules:
        console.print(f"\n[bold]Variation Rules:[/bold] ({len(trap.variation_rules)} rules)")
        for rule in trap.variation_rules:
            console.print(f"  - field: {rule.field}, generator: {rule.generator}")

    console.print()


@app.command()
def validate_traps():
    """Validate all trap YAML files in the library."""
    mgr = _get_trap_manager()
    total = mgr.trap_count
    attack_traps = mgr.load_traps(include_controls=False)
    control_traps = [
        t
        for t in mgr.load_traps(include_controls=True)
        if t.trap_type in ("benign_control", "overly_cautious", "benign_code_control")
    ]

    console.print("\n[bold green]Trap validation complete.[/bold green]")
    console.print(f"  Total traps: {total}")
    console.print(f"  Attack traps: {len(attack_traps)}")
    console.print(f"  Control traps: {len(control_traps)}")
    console.print(f"  Categories: {', '.join(mgr.list_categories())}")
    console.print(f"  Types: {len(mgr.list_trap_types())} distinct types")
    console.print(f"  Difficulties: {', '.join(mgr.list_difficulties())}")

    traps_with_remediation = sum(
        1 for t in mgr.load_traps(include_controls=True) if t.remediation is not None
    )
    console.print(f"  Traps with remediation: {traps_with_remediation}/{total}")

    all_valid = True
    for trap_id in [t.trap_id for t in mgr.load_traps(include_controls=True)]:
        trap = mgr.get_trap(trap_id)
        if not trap.trap_id or not trap.trap_type or not trap.base_task:
            console.print(f"  [red]FAIL: {trap_id} - missing required fields[/red]")
            all_valid = False

    if all_valid:
        console.print("  [green]All traps have required fields.[/green]")
    else:
        raise typer.Exit(code=1)


@app.command()
def run(
    trap_file: Optional[str] = typer.Option(None, "--trap-file", help="Path to trap YAML file"),
    agent_type: str = typer.Option("langchain", "--agent-type", help="Agent harness type"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend"),
    anchor_source: Optional[str] = typer.Option(
        None, "--anchor-source", help="Knowledge base path"
    ),
    report: Optional[str] = typer.Option(None, "--report", help="Report output path"),
):
    """Run general agent evaluation (coming in next iteration)."""
    console.print(
        "[yellow]The 'run' command is not yet implemented."
        " It will evaluate a general agent against traps.[/yellow]"
    )
    raise typer.Exit(code=1)


@app.command()
def run_code(
    trap_file: Optional[str] = typer.Option(None, "--trap-file", help="Path to trap YAML file"),
    agent_type: str = typer.Option("codex", "--agent-type", help="Agent harness type"),
    codebase: Optional[str] = typer.Option(None, "--codebase", help="Codebase path"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend"),
    test_suite: Optional[str] = typer.Option(None, "--test-suite", help="Test suite path"),
    report: Optional[str] = typer.Option(None, "--report", help="Report output path"),
):
    """Run code agent evaluation (coming in next iteration)."""
    console.print(
        "[yellow]The 'run-code' command is not yet implemented."
        " It will evaluate a code agent against traps.[/yellow]"
    )
    raise typer.Exit(code=1)
