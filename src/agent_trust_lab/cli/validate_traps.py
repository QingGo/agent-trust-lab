"""Command: validate_traps"""
import typer

from agent_trust_lab.cli import app, console
from agent_trust_lab.cli._shared import _get_trap_manager


@app.command()
def validate_traps():
    """Validate all trap YAML files in the library."""
    mgr = _get_trap_manager()
    total = mgr.trap_count
    attack_traps = mgr.load_traps(include_controls=False)
    control_traps = [
        t
        for t in mgr.load_traps(include_controls=True)
        if t.trap_type in ("benign_control", "overly_cautious", "benign_code_control")
    ]

    console.print("\n[bold green]Trap validation complete.[/bold green]")
    console.print(f"  Total traps: {total}")
    console.print(f"  Attack traps: {len(attack_traps)}")
    console.print(f"  Control traps: {len(control_traps)}")
    console.print(f"  Categories: {', '.join(mgr.list_categories())}")
    console.print(f"  Types: {len(mgr.list_trap_types())} distinct types")
    console.print(f"  Difficulties: {', '.join(mgr.list_difficulties())}")

    traps_with_remediation = sum(
        1 for t in mgr.load_traps(include_controls=True) if t.remediation is not None
    )
    console.print(f"  Traps with remediation: {traps_with_remediation}/{total}")

    all_valid = True
    for trap_id in [t.trap_id for t in mgr.load_traps(include_controls=True)]:
        trap = mgr.get_trap(trap_id)
        if not trap or not trap.trap_id or not trap.trap_type or not trap.base_task:
            missing = []
            if not trap:
                missing.append("entire trap")
            else:
                if not trap.trap_id:
                    missing.append("trap_id")
                if not trap.trap_type:
                    missing.append("trap_type")
                if not trap.base_task:
                    missing.append("base_task")
            console.print(
                f"  [red]FAIL: {trap_id} - missing required fields: {', '.join(missing)}[/red]"
            )
            all_valid = False

    if all_valid:
        console.print("  [green]All traps have required fields.[/green]")
    else:
        raise typer.Exit(code=1)
