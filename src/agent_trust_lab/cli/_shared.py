"""Shared CLI helpers used across command modules."""

import json
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import console
from agent_trust_lab.config import DEFAULT_MODEL  # noqa: F401


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
    """Load evaluation config from a YAML or JSON file."""
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


def _parse_vote_models(gsar_vote_models: str) -> list:
    """Parse comma-separated model string into a list of model names."""
    return [m.strip() for m in gsar_vote_models.split(",") if m.strip()]


def _build_run_config_params(
    *,
    agent_type: str,
    model: str,
    base_url: str = "",
    sandbox: str = "docker",
    sandbox_image: str = "",
    sandbox_network: bool = False,
    docker_host: str = "",
    skip_hallukg: bool = False,
    thinking_enabled: bool = False,
    reasoning_effort: str = "",
    timeout: int = 120,
    parallel: int = 1,
    max_steps: int = 10,
    output_dir: str = "./output/results/",
    grounded_threshold: float = 0.3,
    nli_weight: float = 0.5,
    injection_template: str = "",
    gsar_vote: bool = True,
    vote_models: list | None = None,
    no_strict: bool = False,
    codebase_path: str | None = None,
) -> dict:
    """Build a shared config_params dict for run/run_code commands."""
    return {
        "agent_type": agent_type,
        "model": model,
        "base_url": base_url,
        "sandbox": sandbox,
        "sandbox_image": sandbox_image,
        "sandbox_network": sandbox_network,
        "docker_host": docker_host,
        "skip_hallukg": skip_hallukg,
        "thinking_enabled": thinking_enabled,
        "reasoning_effort": reasoning_effort,
        "timeout": timeout,
        "parallel": parallel,
        "max_steps": max_steps,
        "output_dir": output_dir,
        "grounded_threshold": grounded_threshold,
        "nli_neutral_weight": nli_weight,
        "injection_template": injection_template,
        "gsar_vote_enabled": gsar_vote,
        "gsar_vote_models": vote_models or [],
        "strict_mode": not no_strict,
        "codebase_path": codebase_path or "",
    }


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


def _display_results(results):
    """Display evaluation results in a table with compliance and hallucination scores."""

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
