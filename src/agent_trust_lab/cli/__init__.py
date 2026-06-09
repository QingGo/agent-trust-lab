"""CLI entry point for agent-trust-lab — 21 Typer commands.

Each command lives in its own module under cli/. Shared helpers (trap
resolution, config building, progress display) live in _shared.py.

Imports of command modules at the bottom register all commands on `app`.
"""

import typer
from rich.console import Console

app = typer.Typer(
    name="agent-trust-lab",
    help="Agent reliability and hallucination evaluation toolkit.",
    no_args_is_help=True,
)

console = Console()

# Import command modules to register them with @app.command()
from agent_trust_lab.cli import (  # noqa: E402, F401
    annotate,
    batch,
    calibrate,
    config,
    diff,
    extract_calibration,
    generate_novel,
    generate_traps,
    harden_traps,
    list_traps,
    perturb,
    rejudge,
    replay,
    report,
    run,
    run_code,
    serve,
    setup_onnx,
    show_trap,
    validate_judge,
    validate_traps,
)
