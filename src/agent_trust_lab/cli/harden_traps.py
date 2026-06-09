"""Command: harden_traps"""
from pathlib import Path
from typing import Optional

import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.cli._shared import _get_traps_data_dir
from agent_trust_lab.config import DEFAULT_MODEL
from agent_trust_lab.log import setup_logging, cli_verbosity_to_level


@app.command()
def harden_traps(
    comparison_path: str = typer.Argument(
        ..., help="Path to comparison.json from multi-model eval"
    ),
    trap_library: Optional[str] = typer.Option(
        None, "--trap-library", help="Path to trap YAML library (default: traps data dir)"
    ),
    output_dir: str = typer.Option(
        "", "--output-dir", "-o", help="Output directory for hardened YAMLs (default: overwrite)"
    ),
    intensity: str = typer.Option(
        "moderate",
        "--intensity",
        help="Hardening intensity: light, moderate, aggressive",
    ),
    max_spread: float = typer.Option(
        0.05,
        "--max-spread",
        help="Max inter-model trust spread to consider hardenable (default: 0.05)",
    ),
    min_max_trust: float = typer.Option(
        0.90,
        "--min-max-trust",
        help="Min max trust (ceiling) to consider hardenable (default: 0.90)",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", "-m", help="LLM model for hardening"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview what would be done without writing files"
    ),
    no_backup: bool = typer.Option(
        False, "--no-backup", help="Skip creating .bak backups of original files"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
):
    """Harden low-discrimination traps from a comparison evaluation.

    Reads a comparison.json from a multi-model evaluation, identifies traps
    where all models score similarly well (high ceiling, near-zero spread),
    and uses LLM rewriting to increase difficulty.

    Examples:
        agent-trust-lab harden-traps results/cmp_3models/comparison.json
        agent-trust-lab harden-traps results/cmp_3models/comparison.json --dry-run
        agent-trust-lab harden-traps results/cmp_3models/comparison.json --intensity aggressive
    """

    from agent_trust_lab.redteam import HardenerConfig, TrapHardener

    setup_logging(level=cli_verbosity_to_level(verbose))

    if not Path(comparison_path).is_file():
        console.print(f"[red]Comparison file not found: {comparison_path}[/red]")
        raise typer.Exit(code=1)

    if intensity not in ("light", "moderate", "aggressive"):
        console.print(
            f"[red]Invalid intensity: {intensity}. "
            "Use light, moderate, or aggressive.[/red]"
        )
        raise typer.Exit(code=1)

    trap_library_path = trap_library or str(_get_traps_data_dir())

    config = HardenerConfig(
        trap_library_path=trap_library_path,
        output_dir=output_dir,
        model=model,
        intensity=intensity,
        backup_originals=not no_backup,
        dry_run=dry_run,
    )

    hardener = TrapHardener(config)

    console.print(f"[bold]Hardening traps from:[/bold] {comparison_path}")
    console.print(
        f"  Intensity: {intensity}, spread < {max_spread}, max trust > {min_max_trust}"
    )
    if dry_run:
        console.print("  [yellow]DRY RUN MODE — no files will be written[/yellow]")

    hardened = hardener.harden_from_comparison(
        comparison_path=comparison_path,
        max_spread=max_spread,
        min_max_trust=min_max_trust,
    )

    if not hardened:
        console.print("[yellow]No hardenable traps found.[/yellow]")
        return

    written = 0
    skipped = 0
    for h in hardened:
        tid = h.get("trap_id", "?")
        tt = h.get("trap_type", "?")
        dif = h.get("difficulty", "?")
        trap_file = Path(trap_library_path) / "general" / f"{tid}.yaml"
        if not output_dir and Path(str(trap_file) + ".bak").exists():
            console.print(f"  [dim]SKIP[/dim] {tid} ({tt}) — already hardened (.bak exists)")
            skipped += 1
            continue
        console.print(f"  {tid} ({tt}) → difficulty={dif}")
        hardener.write_hardened(h)
        written += 1

    console.print(f"\n[green]Hardened: {written} new, Skipped: {skipped} already done[/green]")
