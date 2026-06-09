"""Command: run"""
from typing import List, Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
from agent_trust_lab.cli._shared import (
    _build_run_config_params,
    _display_results,
    _load_config_file,
    _parse_vote_models,
    _run_evaluation,
)
from agent_trust_lab.config import DEFAULT_MODEL


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
    """Run general agent evaluation against traps."""

    setup_logging(level=cli_verbosity_to_level(verbose), log_file=log_file)

    vote_models = _parse_vote_models(gsar_vote_models) if gsar_vote_models else []

    config_params = _load_config_file(config_file) if config_file else {}
    config_params.update(
        _build_run_config_params(
            agent_type=agent_type,
            model=model,
            base_url=base_url or "",
            sandbox=sandbox,
            sandbox_image=sandbox_image or "",
            sandbox_network=sandbox_network,
            docker_host=docker_host or "",
            skip_hallukg=skip_hallukg,
            thinking_enabled=thinking,
            reasoning_effort=effort,
            timeout=timeout,
            parallel=parallel,
            max_steps=max_steps,
            output_dir=output_dir,
            grounded_threshold=grounded_threshold,
            nli_weight=nli_weight,
            injection_template=injection_template,
            gsar_vote=gsar_vote,
            vote_models=vote_models,
            no_strict=no_strict,
        )
    )
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
