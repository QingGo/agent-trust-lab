"""Command: setup_onnx"""
import os
from typing import Optional

import typer
from rich.table import Table

from agent_trust_lab.cli import app, console
from agent_trust_lab.log import cli_verbosity_to_level, setup_logging
from agent_trust_lab.onnx_setup import check_models_available, export_all


@app.command()
def setup_onnx(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Model to export: nli (deberta-base-mnli), embed (all-MiniLM-L6-v2), or all",
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

    Downloads and converts deberta-base-mnli (NLI, ~500MB) and/or
    all-MiniLM-L6-v2 (embedding, ~90MB) from HuggingFace to ONNX format.
    Models are cached at ~/.cache/agent-trust-lab/onnx/.
    """

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
        model_info = {"nli": "NLI (deberta-base-mnli)", "embed": "Embedding (all-MiniLM-L6-v2)"}
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
