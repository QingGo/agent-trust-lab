"""Command: serve"""
import typer

from agent_trust_lab.cli import app, console


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(
        7860, "--port", "-p", help="Port to bind the server to"
    ),
    share: bool = typer.Option(
        False, "--share", help="Create a public shareable link via Gradio"
    ),
):
    """Launch the Gradio web UI for interactive agent evaluation.

    Provides a browser-based interface with:
    - Trap selector (dropdowns for category, type, difficulty)
    - Agent config panel (harness type, model, thinking, sandbox)
    - Real-time trajectory panel with step-by-step rendering
    - Report viewer and download
    """
    from agent_trust_lab.web import launch_ui

    console.print("[bold]Starting Agent Trust Lab UI...[/bold]")
    console.print(f"  URL: http://{host}:{port}")
    if share:
        console.print("  [dim]Public share link will be generated[/dim]")

    launch_ui(server_name=host, server_port=port, share=share)
