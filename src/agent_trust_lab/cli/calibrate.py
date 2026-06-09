"""Command: calibrate"""
import json
from pathlib import Path
from typing import Optional

import typer

from agent_trust_lab.cli import app, console


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
    When --output-dir is provided, generates a calibrated results JSON with recalibrated scores.
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
            "[yellow]Specify --annotations to create a profile, or --output-dir to apply one.[/yellow]"
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

        with open(results_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        calibrated = _apply_calibration_to_results(data, profile)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(calibrated, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Calibrated results saved to {output_json}[/green]")
