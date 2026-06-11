"""Command: replay"""
from pathlib import Path
from typing import Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.cli._shared import (
    _display_results,
    _get_traps_data_dir,
    _load_config_file,
    _parse_vote_models,
)
from agent_trust_lab.config import DEFAULT_MODEL
from agent_trust_lab.log import cli_verbosity_to_level, setup_logging


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
        "./output/results/", "--output-dir", help="Output directory for results"
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
    gsar_vote: bool = typer.Option(
        True, "--gsar-vote/--no-gsar-vote", help="Enable multi-model GSAR voting"
    ),
    gsar_vote_models: Optional[str] = typer.Option(
        "deepseek-v4-flash,deepseek-v4-pro",
        "--gsar-vote-models",
        help="Comma-separated list of models for GSAR voting",
    ),
    no_strict: bool = typer.Option(
        False, "--no-strict", help="Disable strict mode (allow stub fallback for judges)"
    ),
):
    """Replay a captured trajectory through audit and hallucination detection.

    Loads a trajectory JSON file, reconstructs the SecureTrajectory, and runs
    compliance audit + HalluKG evaluation with the specified model config.
    Useful for re-scoring with updated detectors or different model settings.
    """
    import json as _json

    from agent_trust_lab.config import EvaluationConfig
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

    vote_models = (
        _parse_vote_models(gsar_vote_models)
        if gsar_vote_models else []
    )

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
        "gsar_vote_enabled": gsar_vote,
        "gsar_vote_models": vote_models,
        "strict_mode": not no_strict,
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
