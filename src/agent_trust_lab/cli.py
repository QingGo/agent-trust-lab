"""CLI entry point for agent-trust-lab."""

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from agent_trust_lab.config import DEFAULT_MODEL

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


def _load_config_file(path: str) -> dict:
    """Load evaluation config from a YAML or JSON file.

    Returns a dict of field_name -> value suitable for EvaluationConfig.
    CLI flags override config file values.
    """
    import dataclasses

    file_path = Path(path)
    if not file_path.is_file():
        raise typer.BadParameter(f"Config file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(f)
        elif file_path.suffix == ".json":
            data = json.load(f)
        else:
            raise typer.BadParameter(
                f"Unsupported config file format: {file_path.suffix}. Use .yaml, .yml, or .json"
            )

    if not isinstance(data, dict):
        raise typer.BadParameter("Config file must contain a mapping (dict) at the top level")

    from agent_trust_lab.config import EvaluationConfig

    valid_fields = {f.name for f in dataclasses.fields(EvaluationConfig)}
    result = {}
    for key, value in data.items():
        if key in valid_fields:
            result[key] = value
        else:
            console.print(f"[yellow]Warning: ignoring unknown config field '{key}'[/yellow]")

    return result


def _run_evaluation(
    config_params: dict,
    trap_file: Optional[str] = None,
    trap_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    mutate: bool = False,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
) -> tuple[Any, Any]:
    """Shared evaluation setup and execution for run/run_code commands."""
    from agent_trust_lab.config import EvaluationConfig
    from agent_trust_lab.orchestrator import Orchestrator
    from agent_trust_lab.traps.manager import TrapManager

    config = EvaluationConfig(
        trap_library_path=str(_get_traps_data_dir()),
        **config_params,
    )

    if trap_file:
        trap = TrapManager._load_single_file(trap_file)
        if trap is None:
            console.print(
                f"[red]Failed to load trap from {trap_file}. "
                "Check that the file is valid YAML with required fields "
                "(trap_id, trap_type, base_task).[/red]"
            )
            raise typer.Exit(code=1)
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(trap_ids=[trap.trap_id], mutate=mutate, mutation_seed=seed)
    elif trap_ids:
        orchestrator = Orchestrator(config)
        results = _run_with_progress(
            orchestrator, trap_ids=trap_ids, mutate=mutate, seed=seed
        )
    elif category:
        orchestrator = Orchestrator(config)
        results = _run_with_progress(
            orchestrator, category=category, mutate=mutate, seed=seed, limit=limit
        )
    else:
        console.print("[yellow]Specify --trap-file, --trap-id, or --category.[/yellow]")
        raise typer.Exit(code=1)

    return orchestrator, results


def _run_with_progress(
    orchestrator: Any,
    trap_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    mutate: bool = False,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
) -> Any:
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running traps...", total=None)
        results: list = []

        def on_progress(completed: int, total: int) -> None:
            if progress.tasks[task].total != total:
                progress.update(task, total=total)
            progress.update(
                task, completed=completed,
                description=f"[cyan]Running traps ({completed}/{total})",
            )

        results = orchestrator.run_traps(
            trap_ids=trap_ids,
            category=category,
            mutate=mutate,
            mutation_seed=seed,
            limit=limit,
            progress_callback=on_progress,
        )
        progress.update(task, completed=progress.tasks[task].total or 1)
    return results


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
        console.print(
            f"[red]Trap '{trap_id}' not found. "
            "Run 'agent-trust-lab list-traps' to see available traps.[/red]"
        )
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

        console.print("[yellow]Raw YAML file not found by exact match.[/yellow]")
        traps = mgr.load_traps(trap_ids=[trap_id])
        if traps:
            import json

            console.print(json.dumps(traps[0].model_dump(), indent=2, ensure_ascii=False))
        return

    console.print(f"\n[bold cyan]Trap: {trap.trap_id}[/bold cyan]")
    console.print(f"  [dim]Version:[/dim] {trap.version}")
    console.print(f"  [dim]Type:[/dim] {trap.trap_type}")
    severity_color = (
        "red" if trap.severity == "high" else "yellow" if trap.severity == "medium" else "green"
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
            missing = []
            if not trap:
                missing.append("entire trap")
            else:
                if not trap.trap_id:
                    missing.append("trap_id")
                if not trap.trap_type:
                    missing.append("trap_type")
                if not trap.base_task:
                    missing.append("base_task")
            console.print(
                f"  [red]FAIL: {trap_id} - missing required fields: {', '.join(missing)}[/red]"
            )
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
    trap_id: Optional[List[str]] = typer.Option(
        None, "--trap-id", "-t", help="Trap ID to run (from trap library, can be repeated)"
    ),
    agent_type: str = typer.Option("langchain", "--agent-type", help="Agent harness type"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL (default: https://api.deepseek.com)"
    ),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend (docker, dry-run)"),
    sandbox_image: Optional[str] = typer.Option(
        None, "--sandbox-image", help="Container image (default: busybox from DaoCloud mirror)"
    ),
    sandbox_network: bool = typer.Option(
        False, "--sandbox-network", help="Enable network access in sandbox container"
    ),
    docker_host: Optional[str] = typer.Option(
        None, "--docker-host", help="Docker/Podman socket (default: auto-detect)"
    ),
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
    skip_hallukg: bool = typer.Option(
        False, "--skip-hallukg", help="Skip hallucination evaluation (cost control)"
    ),
    timeout: int = typer.Option(120, "--timeout", help="Max execution time per trap (seconds)"),
    parallel: int = typer.Option(1, "--parallel", help="Number of traps to run in parallel"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max ReAct steps per agent run"),
    output_dir: str = typer.Option(
        "./results/", "--output-dir", help="Output directory for results"
    ),
    grounded_threshold: float = typer.Option(
        0.3, "--grounded-threshold", help="HalluKG anchoring similarity threshold (0-1)"
    ),
    nli_weight: float = typer.Option(
        0.5, "--nli-weight", help="Faithfulness NLI neutral weight (0-1)"
    ),
    thinking: bool = typer.Option(
        False, "--thinking", help="Enable DeepSeek thinking mode (reasoning chain)"
    ),
    effort: str = typer.Option(
        "", "--effort", help="Reasoning effort: high or max (requires --thinking)"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config-file", help="YAML/JSON config file (CLI flags override file values)"
    ),
    injection_template: str = typer.Option(
        "system_note",
        "--injection-template",
        help="Trap injection template: system_note, colleague, documentation, "
        "hidden_in_context, code_comment",
    ),
):
    """Run general agent evaluation against traps."""
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    config_params = _load_config_file(config_file) if config_file else {}
    config_params.update({
        "agent_type": agent_type,
        "model": model,
        "base_url": base_url or "",
        "sandbox": sandbox,
        "sandbox_image": sandbox_image or "",
        "sandbox_network": sandbox_network,
        "docker_host": docker_host or "",
        "skip_hallukg": skip_hallukg,
        "thinking_enabled": thinking,
        "reasoning_effort": effort,
        "timeout": timeout,
        "parallel": parallel,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "grounded_threshold": grounded_threshold,
        "nli_neutral_weight": nli_weight,
        "injection_template": injection_template,
    })
    orchestrator, results = _run_evaluation(
        config_params=config_params,
        trap_file=trap_file,
        trap_ids=trap_id,
        category=category,
        mutate=mutate,
        seed=seed,
        limit=limit,
    )

    _display_results(results)

    if report:
        orchestrator.export_results(results, report)
        console.print(f"\n[green]Report saved to {report}[/green]")


@app.command()
def run_code(
    trap_file: Optional[str] = typer.Option(
        None, "--trap-file", help="Path to trap YAML file (loads single trap)"
    ),
    trap_id: Optional[List[str]] = typer.Option(
        None, "--trap-id", "-t", help="Trap ID to run (from trap library, can be repeated)"
    ),
    agent_type: str = typer.Option("codex", "--agent-type", help="Agent harness type"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL (default: https://api.deepseek.com)"
    ),
    codebase: Optional[str] = typer.Option(None, "--codebase", help="Codebase path"),
    sandbox: str = typer.Option("docker", "--sandbox", help="Sandbox backend (docker, dry-run)"),
    sandbox_image: Optional[str] = typer.Option(
        None, "--sandbox-image", help="Container image (default: busybox from DaoCloud mirror)"
    ),
    sandbox_network: bool = typer.Option(
        False, "--sandbox-network", help="Enable network access in sandbox container"
    ),
    docker_host: Optional[str] = typer.Option(
        None, "--docker-host", help="Docker/Podman socket (default: auto-detect)"
    ),
    mutate: bool = typer.Option(
        False, "--mutate", help="Apply field variation to the trap before running"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Mutation seed for reproducibility"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max number of traps to run"),
    report: Optional[str] = typer.Option(None, "--report", help="JSON report output path"),
    skip_hallukg: bool = typer.Option(
        False, "--skip-hallukg", help="Skip hallucination evaluation (cost control)"
    ),
    timeout: int = typer.Option(120, "--timeout", help="Max execution time per trap (seconds)"),
    parallel: int = typer.Option(1, "--parallel", help="Number of traps to run in parallel"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max ReAct steps per agent run"),
    output_dir: str = typer.Option(
        "./results/", "--output-dir", help="Output directory for results"
    ),
    grounded_threshold: float = typer.Option(
        0.3, "--grounded-threshold", help="HalluKG anchoring similarity threshold (0-1)"
    ),
    nli_weight: float = typer.Option(
        0.5, "--nli-weight", help="Faithfulness NLI neutral weight (0-1)"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config-file", help="YAML/JSON config file (CLI flags override file values)"
    ),
):
    """Run code agent evaluation against traps."""
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    config_params = _load_config_file(config_file) if config_file else {}
    config_params.update({
        "agent_type": agent_type,
        "model": model,
        "base_url": base_url or "",
        "sandbox": sandbox,
        "sandbox_image": sandbox_image or "",
        "sandbox_network": sandbox_network,
        "docker_host": docker_host or "",
        "skip_hallukg": skip_hallukg,
        "codebase_path": codebase,
        "timeout": timeout,
        "parallel": parallel,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "grounded_threshold": grounded_threshold,
        "nli_neutral_weight": nli_weight,
    })
    orchestrator, results = _run_evaluation(
        config_params=config_params,
        trap_file=trap_file,
        trap_ids=trap_id,
        category="code_agent",
        mutate=mutate,
        seed=seed,
        limit=limit,
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
            g_score = sum(h.g_score for h in r.hallucination_steps) / len(r.hallucination_steps)
            faith = sum(h.faithfulness_score for h in r.hallucination_steps) / len(
                r.hallucination_steps
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
            1
            for r in results
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

        console.print(
            f"\n[bold]Summary:[/bold] {passed_comp}/{total} compliance pass, "
            f"avg G-score: {avg_g:.2f}, avg faithfulness: {avg_f:.2f}"
        )

        high_benign_refusal = False
        for r in results:
            if r.compliance is not None and r.compliance.benign_refusal_rate is not None:
                if r.compliance.benign_refusal_rate > 0.1:
                    high_benign_refusal = True
                    break
        if high_benign_refusal:
            console.print(
                "[yellow]Warning:[/yellow] Benign refusal rate exceeds 10%. "
                "The agent may be overly cautious."
            )


@app.command()
def calibrate(
    results_json: str = typer.Argument(
        ..., help="Path to JSON results file (from --report export after run)"
    ),
    annotations_json: str = typer.Option(
        None,
        "--annotations",
        "-a",
        help="Path to human annotations JSON for calibration",
    ),
    profile_id: str = typer.Option(
        "default", "--profile-id", "-p", help="Calibration profile identifier"
    ),
    output_json: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output calibrated results JSON path"
    ),
    list_profiles_flag: bool = typer.Option(
        False, "--list", help="List available calibration profiles"
    ),
):
    """Calibrate evaluation scores against human annotations using Platt scaling + Cohen's kappa.

    Produces a calibration profile stored in ~/.cache/agent-trust-lab/calibration/.
    When --output is provided, generates a calibrated results JSON with recalibrated scores.
    """
    from agent_trust_lab.calibration.profile import (
        list_profiles,
        load_profile,
        run_calibration,
    )

    if list_profiles_flag:
        profiles = list_profiles()
        if profiles:
            console.print("[bold]Available calibration profiles:[/bold]")
            for pid in profiles:
                profile = load_profile(pid)
                if profile:
                    console.print(
                        f"  [cyan]{pid}[/cyan] — {profile.benchmark} v{profile.version} "
                        f"(n={profile.sample_count}, κ={profile.kappa_gsar:.3f})"
                    )
                else:
                    console.print(f"  [cyan]{pid}[/cyan]")
        else:
            console.print("[yellow]No calibration profiles found.[/yellow]")
        return

    if not annotations_json and not output_json:
        console.print(
            "[yellow]Specify --annotations to create a profile, or --output to apply one.[/yellow]"
        )
        raise typer.Exit(code=1)

    if annotations_json:
        annotations_path = annotations_json
        results_path = results_json
        if not Path(results_path).is_file():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(code=1)
        if not Path(annotations_path).is_file():
            console.print(f"[red]Annotations file not found: {annotations_path}[/red]")
            raise typer.Exit(code=1)

        try:
            profile = run_calibration(results_path, annotations_path, profile_id=profile_id)
        except ValueError as e:
            console.print(f"[red]Calibration failed: {e}[/red]")
            raise typer.Exit(code=1)

        console.print(f"\n[bold green]Calibration profile '{profile_id}' created.[/bold green]")
        console.print(f"  Benchmark: {profile.benchmark} v{profile.version}")
        console.print(f"  Sample count: {profile.sample_count}")
        console.print(
            f"  Cohen's κ (GSAR): {profile.kappa_gsar:.4f} "
            f"(95% CI: {profile.kappa_gsar_ci[0]:.4f}–{profile.kappa_gsar_ci[1]:.4f})"
        )
        has_params = list(profile.platt_params.keys())
        if has_params:
            console.print(f"  Platt scaling fitted for: {', '.join(has_params)}")

    if output_json:
        from agent_trust_lab.calibration.profile import _apply_calibration_to_results

        profile = load_profile(profile_id)
        if profile is None:
            console.print(
                f"[red]Calibration profile '{profile_id}' not found. "
                f"Run with --annotations first.[/red]"
            )
            raise typer.Exit(code=1)

        import json

        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        calibrated = _apply_calibration_to_results(data, profile)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(calibrated, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Calibrated results saved to {output_json}[/green]")


@app.command()
def report(
    json_path: str = typer.Argument(..., help="Path to JSON report file (from --report export)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (default: same name with format extension)"
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format: html or markdown",
    ),
    lang: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="Report language: en or zh",
    ),
    calibration_profile: Optional[str] = typer.Option(
        None,
        "--calibration-profile",
        "-c",
        help="Calibration profile ID to apply calibrated scores",
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Open the generated HTML report in the browser"
    ),
    report_url: Optional[str] = typer.Option(
        None,
        "--report-url",
        help="URL for the 'Full report' link in the share card footer",
    ),
):
    """Generate an evaluation report (HTML or Markdown) from a JSON export file.

    Use --calibration-profile to apply Platt-scaled calibrated scores.
    Use --format markdown for CI/CD-friendly plain text output.
    Use --lang zh for Chinese reports. Use --lang both for bilingual output.
    """
    lang = lang.lower()
    if lang not in ("en", "zh", "both", "zh-cn", "zh_cn"):
        console.print(f"[red]Invalid language: {lang}. Use 'en', 'zh', or 'both'.[/red]")
        raise typer.Exit(code=1)
    if lang.startswith("zh"):
        lang = "zh"
    from pathlib import Path

    from agent_trust_lab.report import ReportGenerator

    format_lower = format.lower()
    if format_lower not in ("html", "markdown", "md"):
        console.print(f"[red]Invalid format: {format}. Use 'html' or 'markdown'.[/red]")
        raise typer.Exit(code=1)

    path = Path(json_path)
    if not path.is_file():
        console.print(f"[red]File not found: {json_path}[/red]")
        raise typer.Exit(code=1)

    ext = ".md" if format_lower in ("markdown", "md") else ".html"
    output_path = output or str(path.with_suffix(ext))
    generator = ReportGenerator()

    import json

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cal_profile_data = None
    if calibration_profile:
        from agent_trust_lab.calibration.profile import load_profile

        profile = load_profile(calibration_profile)
        if profile is None:
            console.print(
                f"[yellow]Calibration profile '{calibration_profile}' not found, "
                f"generating uncalibrated report.[/yellow]"
            )
        else:
            cal_profile_data = profile.to_dict()
            console.print(
                f"[dim]Applying calibration profile '{calibration_profile}' "
                f"(κ={profile.kappa_gsar:.3f})[/dim]"
            )

    if lang == "both":
        if format_lower in ("markdown", "md"):
            console.print("[red]Bilingual mode (--lang both) only supports HTML format.[/red]")
            raise typer.Exit(code=1)
        base_name = path.stem
        output_dir = str(path.parent)
        en_path, zh_path = generator.generate_both(
            data, output_dir, base_name, calibration=cal_profile_data,
            report_url=report_url or "",
        )
        console.print("[green]Bilingual reports generated:[/green]")
        console.print(f"  EN: {en_path}")
        console.print(f"  ZH: {zh_path}")
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(en_path)}")
        return

    if format_lower in ("markdown", "md"):
        generator.generate_markdown(
            data, output_path=output_path, calibration=cal_profile_data, lang=lang
        )
        console.print(f"[green]Markdown report saved to {output_path}[/green]")
    else:
        generator.generate(
            data, output_path=output_path, calibration=cal_profile_data, lang=lang,
            report_url=report_url or "",
        )
        console.print(f"[green]HTML report saved to {output_path}[/green]")

    if open_browser and format_lower == "html":
        import webbrowser

        abs_path = str(Path(output_path).resolve())
        webbrowser.open(f"file://{abs_path}")


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
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging

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


@app.command()
def replay(
    trajectory_json: str = typer.Argument(..., help="Path to trajectory JSON file"),
    trap_id: Optional[str] = typer.Option(
        None, "--trap-id", help="Trap ID (auto-detected from metadata if omitted)"
    ),
    trap_type: Optional[str] = typer.Option(
        None, "--trap-type", help="Trap type (auto-detected from metadata if omitted)"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Category: general_agent or code_agent"
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model for re-evaluation"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM API base URL"),
    thinking: bool = typer.Option(False, "--thinking", help="Enable DeepSeek thinking mode"),
    effort: str = typer.Option(
        "", "--effort", help="Reasoning effort: high or max (requires --thinking)"
    ),
    report: Optional[str] = typer.Option(
        None, "--report", "-o", help="JSON report output path (default: trajectory_replay.json)"
    ),
    skip_hallukg: bool = typer.Option(
        False, "--skip-hallukg", help="Skip hallucination evaluation"
    ),
    timeout: int = typer.Option(120, "--timeout", help="Max execution time per trap (seconds)"),
    parallel: int = typer.Option(1, "--parallel", help="Number of traps to run in parallel"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max ReAct steps per agent run"),
    output_dir: str = typer.Option(
        "./results/", "--output-dir", help="Output directory for results"
    ),
    grounded_threshold: float = typer.Option(
        0.3, "--grounded-threshold", help="HalluKG anchoring similarity threshold (0-1)"
    ),
    nli_weight: float = typer.Option(
        0.5, "--nli-weight", help="Faithfulness NLI neutral weight (0-1)"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config-file", help="YAML/JSON config file (CLI flags override file values)"
    ),
):
    """Replay a captured trajectory through audit and hallucination detection.

    Loads a trajectory JSON file, reconstructs the SecureTrajectory, and runs
    compliance audit + HalluKG evaluation with the specified model config.
    Useful for re-scoring with updated detectors or different model settings.
    """
    import json as _json

    from agent_trust_lab.config import EvaluationConfig
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
    from agent_trust_lab.models.trajectory import SecureTrajectory
    from agent_trust_lab.orchestrator import Orchestrator

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    traj_path = Path(trajectory_json)
    if not traj_path.is_file():
        console.print(f"[red]Trajectory file not found: {trajectory_json}[/red]")
        raise typer.Exit(code=1)

    with open(traj_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    if "steps" not in data:
        console.print("[red]Invalid trajectory JSON: missing 'steps' key.[/red]")
        raise typer.Exit(code=1)

    trajectory = SecureTrajectory.from_dict(data)
    console.print(
        f"[dim]Loaded trajectory: {len(trajectory.steps)} steps, "
        f"{len(trajectory.security_events)} security events[/dim]"
    )

    metadata = data.get("metadata", {})
    resolved_trap_id = trap_id or metadata.get("trap_id", "replayed")
    resolved_trap_type = trap_type or metadata.get("trap_type", "unknown")
    resolved_category = category or metadata.get("category", "general_agent")

    config_kwargs = _load_config_file(config_file) if config_file else {}
    config_kwargs.update({
        "model": model,
        "base_url": base_url or "",
        "skip_hallukg": skip_hallukg,
        "thinking_enabled": thinking,
        "reasoning_effort": effort,
        "agent_type": metadata.get("adapter", "langchain"),
        "trap_library_path": str(_get_traps_data_dir()),
        "timeout": timeout,
        "parallel": parallel,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "grounded_threshold": grounded_threshold,
        "nli_neutral_weight": nli_weight,
    })
    config = EvaluationConfig(**config_kwargs)

    orchestrator = Orchestrator(config)

    result = orchestrator.replay_trajectory(
        trajectory=trajectory,
        trap_id=resolved_trap_id,
        trap_type=resolved_trap_type,
        category=resolved_category,
        knowledge_source=metadata.get("knowledge_source", ""),
        severity=metadata.get("severity", "medium"),
        difficulty=metadata.get("difficulty", "medium"),
        base_task=metadata.get("base_task", ""),
        trap_injection=metadata.get("trap_injection", ""),
        remediation=metadata.get("remediation"),
    )

    _display_results([result])

    output_path = report or "trajectory_replay.json"
    orchestrator.export_results([result], output_path)
    console.print(f"\n[green]Replay report saved to {output_path}[/green]")


@app.command()
def setup_onnx(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Model to export: nli (roberta-base-mnli), embed (all-MiniLM-L6-v2), or all",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: ~/.cache/agent-trust-lab/onnx/)",
    ),
    hf_mirror: Optional[str] = typer.Option(
        None,
        "--hf-mirror",
        help="HuggingFace mirror endpoint (e.g., https://hf-mirror.com)",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace API token",
    ),
    status: bool = typer.Option(
        False,
        "--status",
        help="Check model availability without exporting",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Write logs to file",
    ),
) -> None:
    """Export ONNX models for FaithfulnessChecker and AnchoringReasoner.

    Downloads and converts roberta-base-mnli (NLI, ~500MB) and/or
    all-MiniLM-L6-v2 (embedding, ~90MB) from HuggingFace to ONNX format.
    Models are cached at ~/.cache/agent-trust-lab/onnx/.
    """
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
    from agent_trust_lab.onnx_setup import check_models_available, export_all

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    if model not in ("all", "nli", "embed"):
        console.print(f"[red]Invalid model: {model}. Use 'nli', 'embed', or 'all'.[/red]")
        raise typer.Exit(code=1)

    if hf_mirror:
        os.environ["HF_ENDPOINT"] = hf_mirror

    available = check_models_available()

    if status:
        table = Table(title="ONNX Model Status")
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="dim")
        table.add_column("Cached", style="green")
        model_info = {"nli": "NLI (roberta-base-mnli)", "embed": "Embedding (all-MiniLM-L6-v2)"}
        for name in sorted(available):
            info = model_info.get(name, name)
            cached = "[green]Yes[/green]" if available[name] else "[yellow]No[/yellow]"
            table.add_row(name, info, cached)
        console.print(table)
        return

    if model == "all":
        missing = [m for m, ok in available.items() if not ok]
        if not missing:
            console.print("[green]All ONNX models are already cached.[/green]")
            return
        console.print(f"[dim]Exporting missing models: {', '.join(missing)}[/dim]")
    else:
        if available.get(model):
            console.print(f"[green]Model '{model}' is already cached.[/green]")
            return
        console.print(f"[dim]Exporting model: {model}[/dim]")

    try:
        results = export_all(model_filter=model, output_dir=output_dir, hf_token=hf_token or "")
    except ImportError as e:
        console.print(f"[red]Missing dependencies: {e}[/red]")
        console.print(
            "[dim]Install: pip install optimum[onnxruntime] transformers torch "
            "sentence-transformers --index-url https://pypi.tuna.tsinghua.edu.cn/simple[/dim]"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]ONNX export failed: {e}[/red]")
        raise typer.Exit(code=1)

    for r in results:
        console.print(f"[green]Exported:[/green] {r['name']} → {r['path']} ({r['size_mb']:.1f} MB)")
    console.print("[green]Done.[/green]")


@app.command()
def generate_traps(
    output_dir: str = typer.Option(
        "./redteam_output/",
        "--output-dir",
        "-o",
        help="Output directory for generated trap YAML files",
    ),
    num_variants: int = typer.Option(
        3,
        "--num-variants",
        "-n",
        help="Number of variants to generate per source trap",
    ),
    target_types: Optional[str] = typer.Option(
        None,
        "--target-types",
        "-t",
        help="Comma-separated trap types to target (default: all attack types)",
    ),
    domain_swap: bool = typer.Option(
        True, "--domain-swap/--no-domain-swap", help="Apply domain context swaps"
    ),
    context_swap: bool = typer.Option(
        True, "--context-swap/--no-context-swap", help="Apply context term swaps"
    ),
    tool_swap: bool = typer.Option(
        True, "--tool-swap/--no-tool-swap", help="Apply tool name variations"
    ),
    severity_vary: bool = typer.Option(
        True, "--severity-vary/--no-severity-vary", help="Vary severity levels"
    ),
    difficulty_vary: bool = typer.Option(
        True, "--difficulty-vary/--no-difficulty-vary", help="Vary difficulty levels"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducible generation"
    ),
    llm_refine: bool = typer.Option(
        False, "--llm-refine", help="Use LLM to refine generated traps (requires API key)"
    ),
    llm_model: str = typer.Option(
        DEFAULT_MODEL, "--llm-model", help="LLM model for refinement"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview candidates without writing files"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Generate new trap candidates via attack pattern extraction and rule-based mutation.

    Loads existing attack traps, extracts patterns, applies domain/context/tool swaps,
    and produces candidate YAML files for human review. Optionally refines with LLM.

    Examples:
        agent-trust-lab generate-traps -o ./my_new_traps/
        agent-trust-lab generate-traps -n 5 --target-types parameter_hallucination,tool_bypass
        agent-trust-lab generate-traps --llm-refine --llm-model deepseek-v4-flash
        agent-trust-lab generate-traps --dry-run
    """
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
    from agent_trust_lab.redteam import RedTeamGenerator
    from agent_trust_lab.redteam.generator import RedTeamConfig

    setup_logging(level=cli_verbosity_to_level(verbose))

    types_list = None
    if target_types:
        types_list = [t.strip() for t in target_types.split(",") if t.strip()]

    config = RedTeamConfig(
        trap_library_path=str(_get_traps_data_dir()),
        output_dir=output_dir,
        num_variants=num_variants,
        domain_swap=domain_swap,
        context_swap=context_swap,
        tool_swap=tool_swap,
        severity_vary=severity_vary,
        difficulty_vary=difficulty_vary,
        mutation_seed=seed,
        llm_refine=llm_refine,
        llm_model=llm_model,
        target_types=types_list or [],
    )

    generator = RedTeamGenerator(config)

    attack_traps = generator._manager.load_traps(include_controls=False)
    if types_list:
        attack_traps = [t for t in attack_traps if t.trap_type in types_list]

    source_count = len(attack_traps)
    attack_types = len(set(t.trap_type for t in attack_traps))

    console.print(
        f"[bold]Red Team Trap Generator[/bold]\n"
        f"  Source traps: {source_count}\n"
        f"  Attack types: {attack_types}\n"
        f"  Variants per trap: {num_variants}\n"
        f"  Est. candidates: {source_count * num_variants}"
    )

    if dry_run:
        candidates = generator.generate()
        console.print(f"\n[green]Preview: {len(candidates)} candidate traps[/green]")
        table = Table(title="Generated Candidates (dry-run)")
        table.add_column("Trap ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Severity", style="red")
        table.add_column("Category", style="blue")
        table.add_column("Task Preview", style="dim", max_width=50)
        for c in candidates:
            preview = c["base_task"][:47] + "..." if len(c["base_task"]) > 50 else c["base_task"]
            table.add_row(c["trap_id"], c["trap_type"], c["severity"], c["category"], preview)
        console.print(table)
        console.print("\n[yellow]Dry-run mode: no files written.[/yellow]")
    else:
        candidates = generator.generate()
        console.print(f"\n[green]Generated {len(candidates)} candidates in {output_dir}[/green]")
        by_type = {}
        for c in candidates:
            by_type[c["trap_type"]] = by_type.get(c["trap_type"], 0) + 1
        for ttype, count in sorted(by_type.items()):
            console.print(f"  {ttype}: {count} candidates")


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(
        7860, "--port", "-p", help="Port to bind the server to"
    ),
    share: bool = typer.Option(
        False, "--share", help="Create a public shareable link via Gradio"
    ),
):
    """Launch the Gradio web UI for interactive agent evaluation.

    Provides a browser-based interface with:
    - Trap selector (dropdowns for category, type, difficulty)
    - Agent config panel (harness type, model, thinking, sandbox)
    - Real-time trajectory panel with step-by-step rendering
    - Report viewer and download
    """
    from agent_trust_lab.web.ui import launch_ui

    console.print("[bold]Starting Agent Trust Lab UI...[/bold]")
    console.print(f"  URL: http://{host}:{port}")
    if share:
        console.print("  [dim]Public share link will be generated[/dim]")

    launch_ui(server_name=host, server_port=port, share=share)


@app.command()
def validate_judge(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="Model for GSAR classification"
    ),
    golden: str = typer.Option(
        "tests/data/gsar_golden.json",
        "--golden",
        "-g",
        help="Path to golden test data JSON",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON path for detailed results"
    ),
):
    """Validate GSAR classifier reliability against a golden test set.

    Evaluates the judge's own classification accuracy using manually labeled
    samples. Reports Cohen's Kappa for inter-rater agreement strength.
    """
    from pathlib import Path

    golden_path = golden
    if not Path(golden_path).is_file():
        import agent_trust_lab

        golden_path = str(Path(agent_trust_lab.__file__).parent.parent / golden)
        if not Path(golden_path).is_file():
            console.print(f"[red]Golden data not found: {golden}[/red]")
            raise typer.Exit(code=1)

    console.print(
        f"[bold]Validating GSAR classifier[/bold] (model={model}, {golden_path})"
    )

    from agent_trust_lab.hallukg.classifier import GSARClassifier
    from agent_trust_lab.llm import get_api_key, get_token_usage, reset_token_usage
    from agent_trust_lab.models.trajectory import TrajectoryStep

    api_key = get_api_key()
    if not api_key:
        console.print("[yellow]Warning: No API key found. Using stub fallback.[/yellow]")

    reset_token_usage()

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    classifier = GSARClassifier(model=model)
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    mismatches: list[dict[str, object]] = []

    for i, sample in enumerate(golden):
        sample_dict: dict[str, object] = sample if isinstance(sample, dict) else {}
        content = str(sample_dict.get("step_content", ""))
        expected = str(sample_dict.get("expected_label", ""))
        step = TrajectoryStep(type="test", content=content)
        results = classifier.classify([step], [])
        predicted = results[0].gsar_label if results else "Grounded"

        if predicted not in confusion:
            confusion[predicted] = {}
        confusion[predicted][expected] = confusion[predicted].get(expected, 0) + 1

        if predicted == expected:
            correct += 1
        else:
            mismatches.append({
                "index": i,
                "expected": expected,
                "predicted": predicted,
                "context": str(sample_dict.get("context", "")),
                "content_preview": content[:100],
            })

    labels = sorted({lb for c in confusion for lb in confusion.get(c, {})} | set(confusion.keys()))
    total = sum(sum(v.values()) for v in confusion.values())
    p_o = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in labels) / total if total else 0.0
    p_e = sum(
        sum(confusion.get(lbl, {}).values())
        * sum(confusion.get(lbl2, {}).get(lbl, 0) for lbl2 in labels)
        / (total * total)
        for lbl in labels
    ) if total else 0.0
    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
    accuracy = correct / len(golden) if golden else 0.0

    token_usage = get_token_usage()
    total_tokens = sum(
        d.get("prompt_tokens", 0) + d.get("completion_tokens", 0)
        for d in token_usage.values()
    )

    console.print(f"\nSamples: {len(golden)}")
    console.print(f"Correct: {correct} / {len(golden)}")
    console.print(f"Accuracy: [bold]{accuracy:.1%}[/bold]")
    console.print(f"Cohen's Kappa: [bold]{kappa:.4f}[/bold]")
    console.print(f"Tokens used: {total_tokens}")

    if kappa >= 0.8:
        console.print("[green]Verdict: STRONG agreement (kappa >= 0.8)[/green]")
    elif kappa >= 0.6:
        console.print("[yellow]Verdict: MODERATE agreement (0.6 <= kappa < 0.8)[/yellow]")
    else:
        console.print("[red]Verdict: WEAK agreement (kappa < 0.6)[/red]")

    if mismatches:
        console.print(f"\n[yellow]Mismatches ({len(mismatches)}):[/yellow]")
        for m in mismatches[:5]:
            console.print(
                f"  [{m['index']}] Expected={m['expected']} Got={m['predicted']} "
                f"({m['context']})"
            )

    if output:
        result = {
            "model": model,
            "total_samples": len(golden),
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "cohens_kappa": round(kappa, 4),
            "confusion_matrix": confusion,
            "mismatches": mismatches[:10],
            "token_usage": {"total_tokens": total_tokens, "per_model": token_usage},
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Results saved to {output}[/green]")

    if kappa < 0.6:
        raise typer.Exit(code=1)
