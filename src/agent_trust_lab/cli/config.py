"""Command: config — manage evaluation configuration."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
import yaml

from agent_trust_lab.cli import app, console
from agent_trust_lab.config import DEFAULT_MODEL, EvaluationConfig


def _config_to_yaml_dict() -> dict:
    """Convert EvaluationConfig defaults to a YAML-friendly dict."""
    cfg = EvaluationConfig()
    raw = asdict(cfg)
    # Remove None values and empty optional fields for cleaner output
    return {k: v for k, v in raw.items() if v is not None and v != [] and v != ""}


def _field_help() -> dict:
    """Return help text for key config fields."""
    return {
        "agent_type": "Agent harness type (langchain, codex, openai, opencode, docker, dry-run)",
        "model": "Model used by the agent under test",
        "judge_model": "Model used by the GSAR judge (defaults to agent model)",
        "sandbox": "Sandbox backend (docker or dry-run)",
        "sandbox_image": "Container image for sandbox execution",
        "sandbox_network": "Allow network access in sandbox containers",
        "timeout": "Per-trap timeout in seconds",
        "parallel": "Number of traps to run concurrently",
        "max_steps": "Maximum agent steps per trap",
        "thinking_enabled": "Enable reasoning chains (DeepSeek thinking mode)",
        "reasoning_effort": "Reasoning effort level (high or max)",
        "temperature": "LLM temperature [0.0, 2.0]",
        "grounded_threshold": "Minimum semantic similarity for grounded classification",
        "nli_neutral_weight": "Weight for neutral class in ONNX NLI score",
        "anchor_type_weights": "α weights for GSAR-NLI fusion per anchor type",
        "gsar_vote_enabled": "Enable multi-model GSAR voting",
        "gsar_vote_models": "Models used for GSAR voting (at least 2)",
        "skip_hallukg": "Skip hallucination KG analysis (faster, less detailed)",
        "adaptive_sampling": "Re-sample steps with high GSAR-NLI disagreement",
        "self_consistency_enabled": "Run multiple GSAR classifications per step",
        "self_consistency_samples": "Number of GSAR samples for self-consistency",
        "cache_enabled": "Cache evaluation results to disk",
        "cache_ttl_days": "Cache time-to-live in days",
        "runs": "Number of evaluation runs per trap (for stability)",
        "strict_mode": "Fail on API errors instead of falling back to stub",
    }


@app.command()
def config(
    init: bool = typer.Option(
        False, "--init", help="Generate a config.yaml with default values"
    ),
    show: Optional[Path] = typer.Option(
        None, "--show", exists=True, help="Display values from a config file"
    ),
    output: Path = typer.Option(
        Path("config.yaml"),
        "--output",
        "-o",
        help="Output file path (used with --init)",
    ),
) -> None:
    """Manage evaluation configuration.

    Without arguments, prints the current default configuration.
    Use --init to generate a config.yaml file from defaults.
    Use --show <config.yaml> to inspect an existing configuration file.
    """
    if show:
        _show_config(show)
        return

    if init:
        _generate_config(output)
        return

    _show_defaults()


def _show_defaults() -> None:
    """Display current default configuration values."""
    cfg_dict = _config_to_yaml_dict()
    help_text = _field_help()

    console.print("\n[bold]EvaluationConfig — Default Values[/bold]\n")

    groups = {
        "Core": [
            "agent_type", "model", "judge_model", "sandbox", "sandbox_image",
            "timeout", "parallel", "max_steps", "runs",
        ],
        "Thinking": ["thinking_enabled", "reasoning_effort", "temperature"],
        "Hallukg": [
            "grounded_threshold", "nli_neutral_weight", "anchor_type_weights",
            "gsar_vote_enabled", "gsar_vote_models", "skip_hallukg",
            "adaptive_sampling", "adaptive_disagreement_threshold", "adaptive_max_samples",
            "self_consistency_enabled", "self_consistency_samples",
        ],
        "Sandbox": ["sandbox_network", "sandbox_tmpfs_size", "docker_host"],
        "Caching": ["cache_enabled", "cache_ttl_days", "cache_dir"],
        "Other": ["skip_extract_types", "strict_mode", "difficulty_weights"],
    }

    for group_name, keys in groups.items():
        console.print(f"  [bold cyan]{group_name}[/bold cyan]")
        for key in keys:
            if key in cfg_dict:
                val = cfg_dict[key]
                hint = help_text.get(key, "")
                console.print(f"    [green]{key}[/green] = {_format_value(val)}")
                if hint:
                    console.print(f"      [dim]{hint}[/dim]")
        console.print()


def _generate_config(output: Path) -> None:
    """Write default config to a YAML file."""
    cfg_dict = _config_to_yaml_dict()
    content = yaml.dump(cfg_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)
    output.write_text("# Agent Trust Lab Configuration\n" + content, encoding="utf-8")
    console.print(f"[green]Config written to {output}[/green]")
    console.print(f"Use: [bold]agent-trust-lab run --config-file {output}[/bold]")


def _show_config(path: Path) -> None:
    """Display config values from a YAML or JSON file."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        console.print(f"[red]Failed to parse {path} as YAML/JSON[/red]")
        raise typer.Exit(1)

    if not isinstance(data, dict):
        console.print("[red]Config file must contain a mapping at top level[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Config from {path}[/bold]\n")
    for key, val in data.items():
        if key in _field_help():
            console.print(f"  [green]{key}[/green] = {_format_value(val)}")
            console.print(f"    [dim]{_field_help()[key]}[/dim]")

    # Warn on unknown keys
    unknown = set(data.keys()) - set(_field_help().keys())
    if unknown:
        console.print(f"\n  [yellow]Unknown keys: {', '.join(sorted(unknown))}[/yellow]")


def _format_value(val) -> str:
    """Format a config value for display."""
    if isinstance(val, dict):
        items = ", ".join(f"{k}={v}" for k, v in val.items())
        return f"{{{items}}}"
    if isinstance(val, list):
        return f"[{', '.join(str(v) for v in val)}]"
    if isinstance(val, bool):
        return f"[bold]{val}[/bold]"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)
