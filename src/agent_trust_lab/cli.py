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

    return Path(str(importlib.resources.files("agent_trust_lab.traps"))) / "data"


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
        if not trap or not trap.trap_id or not trap.trap_type or not trap.base_task:
            console.print(f"  [red]FAIL: {trap_id} - missing required fields[/red]")
            all_valid = False

    if all_valid:
        console.print("  [green]All traps have required fields.[/green]")
    else:
        raise typer.Exit(code=1)


@app.command()
def run(
    trap_file: Optional[str] = typer.Option(
        None, "--trap-file", help="Path to trap YAML file (loads single trap)"
    ),
    trap_id: Optional[str] = typer.Option(
        None, "--trap-id", help="Trap ID to run (from trap library)"
    ),
    agent_type: str = typer.Option("langchain", "--agent-type", help="Agent harness type"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend (docker, dry-run)"),
    mutate: bool = typer.Option(
        False, "--mutate", help="Apply field variation to the trap before running"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Mutation seed for reproducibility"),
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Run all traps in category (general_agent, code_agent)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Max number of traps to run (with --category)"
    ),
    report: Optional[str] = typer.Option(None, "--report", help="JSON report output path"),
):
    """Run general agent evaluation against traps."""
    from agent_trust_lab.config import EvaluationConfig
    from agent_trust_lab.orchestrator import Orchestrator
    from agent_trust_lab.traps.manager import TrapManager

    config = EvaluationConfig(
        agent_type=agent_type,
        model=model,
        sandbox=sandbox,
        trap_library_path=str(_get_traps_data_dir()),
    )

    if trap_file:
        trap = TrapManager._load_single_file(trap_file)
        if trap is None:
            console.print(f"[red]Failed to load trap from {trap_file}[/red]")
            raise typer.Exit(code=1)
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            trap_ids=[trap.trap_id], mutate=mutate, mutation_seed=seed
        )
    elif trap_id:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            trap_ids=[trap_id], mutate=mutate, mutation_seed=seed
        )
    elif category:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            category=category, mutate=mutate, mutation_seed=seed, limit=limit
        )
    else:
        console.print("[yellow]Specify --trap-file, --trap-id, or --category.[/yellow]")
        raise typer.Exit(code=1)

    _display_results(results)

    if report:
        orchestrator.export_results(results, report)
        console.print(f"\n[green]Report saved to {report}[/green]")


@app.command()
def run_code(
    trap_file: Optional[str] = typer.Option(
        None, "--trap-file", help="Path to trap YAML file (loads single trap)"
    ),
    trap_id: Optional[str] = typer.Option(
        None, "--trap-id", help="Trap ID to run (from trap library)"
    ),
    agent_type: str = typer.Option("codex", "--agent-type", help="Agent harness type"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    codebase: Optional[str] = typer.Option(None, "--codebase", help="Codebase path"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend (docker, dry-run)"),
    mutate: bool = typer.Option(
        False, "--mutate", help="Apply field variation to the trap before running"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Mutation seed for reproducibility"),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Max number of traps to run"
    ),
    report: Optional[str] = typer.Option(None, "--report", help="JSON report output path"),
):
    """Run code agent evaluation against traps."""
    from agent_trust_lab.config import EvaluationConfig
    from agent_trust_lab.orchestrator import Orchestrator
    from agent_trust_lab.traps.manager import TrapManager

    config = EvaluationConfig(
        agent_type=agent_type,
        model=model,
        sandbox=sandbox,
        codebase_path=codebase,
        trap_library_path=str(_get_traps_data_dir()),
    )

    if trap_file:
        trap = TrapManager._load_single_file(trap_file)
        if trap is None:
            console.print(f"[red]Failed to load trap from {trap_file}[/red]")
            raise typer.Exit(code=1)
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            trap_ids=[trap.trap_id], mutate=mutate, mutation_seed=seed
        )
    elif trap_id:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            trap_ids=[trap_id], mutate=mutate, mutation_seed=seed
        )
    else:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            category="code_agent", mutate=mutate, mutation_seed=seed, limit=limit
        )

    _display_results(results)

    if report:
        orchestrator.export_results(results, report)
        console.print(f"\n[green]Report saved to {report}[/green]")


def _display_results(results):
    """Display evaluation results in a table with compliance and hallucination scores."""
    from rich.table import Table

    table = Table(title=f"Evaluation Results ({len(results)} traps)")
    table.add_column("Trap ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Steps", style="yellow")
    table.add_column("Compliance", style="red")
    table.add_column("Hallu G", style="magenta")
    table.add_column("Faith", style="magenta")
    table.add_column("Mutated", style="dim")

    for r in results:
        compliance_str = "-"
        if r.compliance is not None:
            status = r.compliance.overall_status()
            color_map = {"pass": "green", "warn": "yellow", "fail": "red"}
            c = color_map.get(status, "white")
            compliance_str = f"[{c}]{status}[/{c}]"

        g_score = 0.0
        faith = 0.0
        if r.hallucination_steps:
            g_score = (
                sum(h.g_score for h in r.hallucination_steps)
                / len(r.hallucination_steps)
            )
            faith = (
                sum(h.faithfulness_score for h in r.hallucination_steps)
                / len(r.hallucination_steps)
            )

        table.add_row(
            r.trap_id,
            r.trap_type,
            r.category,
            str(len(r.trajectory.steps)),
            compliance_str,
            f"{g_score:.2f}",
            f"{faith:.2f}",
            "yes" if r.mutated else "no",
        )

    console.print(table)

    total = len(results)
    if total > 0:
        passed_comp = sum(
            1 for r in results
            if r.compliance is not None and r.compliance.overall_status() == "pass"
        )
        avg_g = 0.0
        avg_f = 0.0
        count_hallu = sum(1 for r in results if r.hallucination_steps)
        if count_hallu > 0:
            hallu_results = [r for r in results if r.hallucination_steps]
            avg_g = sum(
                sum(h.g_score for h in r.hallucination_steps) / len(r.hallucination_steps)
                for r in hallu_results
            ) / len(hallu_results)
            avg_f = sum(
                sum(h.faithfulness_score for h in r.hallucination_steps)
                / len(r.hallucination_steps)
                for r in hallu_results
            ) / len(hallu_results)

        console.print(f"\n[bold]Summary:[/bold] {passed_comp}/{total} compliance pass, "
                      f"avg G-score: {avg_g:.2f}, avg faithfulness: {avg_f:.2f}")
