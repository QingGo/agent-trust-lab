"""CLI entry point for agent-trust-lab."""

import os
from pathlib import Path
from typing import Any, Optional

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


def _run_evaluation(
    config_params: dict,
    trap_file: Optional[str] = None,
    trap_id: Optional[str] = None,
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
            console.print(f"[red]Failed to load trap from {trap_file}[/red]")
            raise typer.Exit(code=1)
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(trap_ids=[trap.trap_id], mutate=mutate, mutation_seed=seed)
    elif trap_id:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(trap_ids=[trap_id], mutate=mutate, mutation_seed=seed)
    elif category:
        orchestrator = Orchestrator(config)
        results = orchestrator.run_traps(
            category=category, mutate=mutate, mutation_seed=seed, limit=limit
        )
    else:
        console.print("[yellow]Specify --trap-file, --trap-id, or --category.[/yellow]")
        raise typer.Exit(code=1)

    return orchestrator, results


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
    model: str = typer.Option("deepseek-v4-flash", "--model", help="LLM model to use"),
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
):
    """Run general agent evaluation against traps."""
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    orchestrator, results = _run_evaluation(
        config_params={
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
        },
        trap_file=trap_file,
        trap_id=trap_id,
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
    trap_id: Optional[str] = typer.Option(
        None, "--trap-id", help="Trap ID to run (from trap library)"
    ),
    agent_type: str = typer.Option("codex", "--agent-type", help="Agent harness type"),
    model: str = typer.Option("deepseek-v4-flash", "--model", help="LLM model to use"),
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
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
    ),
):
    """Run code agent evaluation against traps."""
    from agent_trust_lab.log import cli_verbosity_to_level, setup_logging

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    orchestrator, results = _run_evaluation(
        config_params={
            "agent_type": agent_type,
            "model": model,
            "base_url": base_url or "",
            "sandbox": sandbox,
            "sandbox_image": sandbox_image or "",
            "sandbox_network": sandbox_network,
            "docker_host": docker_host or "",
            "skip_hallukg": skip_hallukg,
            "codebase_path": codebase,
        },
        trap_file=trap_file,
        trap_id=trap_id,
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
        console.print(f"  Cohen's κ (GSAR): {profile.kappa_gsar:.4f} "
                      f"(95% CI: {profile.kappa_gsar_ci[0]:.4f}–{profile.kappa_gsar_ci[1]:.4f})")
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
):
    """Generate an evaluation report (HTML or Markdown) from a JSON export file.

    Use --calibration-profile to apply Platt-scaled calibrated scores.
    Use --format markdown for CI/CD-friendly plain text output.
    Use --lang zh for Chinese reports.
    """
    lang = lang.lower()
    if lang not in ("en", "zh", "zh-cn", "zh_cn"):
        console.print(f"[red]Invalid language: {lang}. Use 'en' or 'zh'.[/red]")
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

    if format_lower in ("markdown", "md"):
        generator.generate_markdown(data, output_path=output_path,
                                    calibration=cal_profile_data, lang=lang)
        console.print(f"[green]Markdown report saved to {output_path}[/green]")
    else:
        generator.generate(data, output_path=output_path,
                           calibration=cal_profile_data, lang=lang)
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

    if batch_config.report_lang.startswith("zh"):
        batch_config.report_lang = "zh"

    fmt = batch_config.report_format
    if fmt not in ("html", "markdown", "md"):
        console.print(f"[red]Invalid report format: {fmt}. Use 'html' or 'markdown'.[/red]")
        raise typer.Exit(code=1)

    ev_labels = ", ".join(s.label for s in batch_config.evaluations)
    console.print(f"[bold]Batch evaluation:[/bold] {ev_labels}")
    console.print(f"  {len(batch_config.evaluations)} config(s), "
                  f"output dir: {batch_config.output_dir}")
    console.print(f"  Report: {batch_config.report_format} ({batch_config.report_lang})")

    merged = run_batch(batch_config)

    n_traps = len(merged.get("results", []))
    n_models = len(merged.get("configs", []))
    console.print(f"\n[green]Batch complete.[/green] "
                  f"{n_traps} traps x {n_models} models."
                  f"\n  Output: {batch_config.output_dir}")


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
    model: str = typer.Option("deepseek-v4-flash", "--model", help="LLM model for re-evaluation"),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL"
    ),
    thinking: bool = typer.Option(
        False, "--thinking", help="Enable DeepSeek thinking mode"
    ),
    effort: str = typer.Option(
        "", "--effort", help="Reasoning effort: high or max (requires --thinking)"
    ),
    report: Optional[str] = typer.Option(
        None, "--report", "-o", help="JSON report output path (default: trajectory_replay.json)"
    ),
    skip_hallukg: bool = typer.Option(
        False, "--skip-hallukg", help="Skip hallucination evaluation"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
    log_file: Optional[str] = typer.Option(
        None, "--log-file", help="Write logs to file instead of stderr"
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
    console.print(f"[dim]Loaded trajectory: {len(trajectory.steps)} steps, "
                  f"{len(trajectory.security_events)} security events[/dim]")

    metadata = data.get("metadata", {})
    resolved_trap_id = trap_id or metadata.get("trap_id", "replayed")
    resolved_trap_type = trap_type or metadata.get("trap_type", "unknown")
    resolved_category = category or metadata.get("category", "general_agent")

    config = EvaluationConfig(
        model=model,
        base_url=base_url or "",
        skip_hallukg=skip_hallukg,
        thinking_enabled=thinking,
        reasoning_effort=effort,
        agent_type=metadata.get("adapter", "langchain"),
        trap_library_path=str(_get_traps_data_dir()),
    )

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
