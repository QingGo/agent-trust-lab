"""Command: report"""
import os
from typing import Optional

import typer

from agent_trust_lab.cli import app, console


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
