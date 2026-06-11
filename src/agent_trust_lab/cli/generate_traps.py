"""Command: generate_traps"""
from typing import Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console
from agent_trust_lab.cli._shared import _get_traps_data_dir
from agent_trust_lab.config import DEFAULT_MODEL
from agent_trust_lab.log import cli_verbosity_to_level, setup_logging


@app.command()
def generate_traps(
    output_dir: str = typer.Option(
        "./output/redteam/",
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
