"""Smoke tests: verify all 21 CLI commands register without crashing."""


def test_all_commands_registered():
    """Verify all 21 CLI commands are registered on the Typer app."""
    from agent_trust_lab.cli import app

    commands = [
        c.name or (c.callback.__name__ if c.callback else None)
        for c in app.registered_commands
    ]
    expected = {
        "annotate",
        "batch",
        "calibrate",
        "config",
        "diff",
        "extract_calibration_data",
        "generate_novel",
        "generate_traps",
        "harden_traps",
        "list_traps",
        "perturb",
        "rejudge",
        "replay",
        "report",
        "run",
        "run_code",
        "serve",
        "setup_onnx",
        "show_trap",
        "validate_judge",
        "validate_traps",
    }
    actual = set(commands)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing commands: {missing}"
    assert not extra, f"Unexpected commands: {extra}"


def test_app_help_no_crash():
    """Verify --help flag does not crash."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list-traps" in result.stdout
    assert "run" in result.stdout


def test_list_traps_no_crash():
    """Verify list-traps command does not crash."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["list-traps"])
    assert result.exit_code == 0


def test_validate_traps_no_crash():
    """Verify validate-traps command does not crash."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate-traps"])
    assert result.exit_code == 0


def test_setup_onnx_status_no_crash():
    """Verify setup-onnx --status does not crash."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["setup-onnx", "--status"])
    assert result.exit_code == 0


def test_config_defaults():
    """Verify config command shows defaults without error."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "EvaluationConfig" in result.stdout
    assert "model" in result.stdout


def test_diff_missing_files():
    """Verify diff exits cleanly with missing file error."""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["diff", "/nonexistent/a.json", "/nonexistent/b.json"])
    assert result.exit_code == 1


# ── --help smoke tests for remaining commands ──

def test_show_trap_help():
    """show-trap --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["show-trap", "--help"])
    assert result.exit_code == 0


def test_run_help():
    """run --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_run_code_help():
    """run-code --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run-code", "--help"])
    assert result.exit_code == 0


def test_replay_help():
    """replay --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["replay", "--help"])
    assert result.exit_code == 0


def test_batch_help():
    """batch --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0


def test_report_help():
    """report --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0


def test_calibrate_help():
    """calibrate --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0


def test_generate_traps_help():
    """generate-traps --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["generate-traps", "--help"])
    assert result.exit_code == 0


def test_harden_traps_help():
    """harden-traps --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["harden-traps", "--help"])
    assert result.exit_code == 0


def test_generate_novel_help():
    """generate-novel --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["generate-novel", "--help"])
    assert result.exit_code == 0


def test_validate_judge_help():
    """validate-judge --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate-judge", "--help"])
    assert result.exit_code == 0


def test_rejudge_help():
    """rejudge --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["rejudge", "--help"])
    assert result.exit_code == 0


def test_perturb_help():
    """perturb --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["perturb", "--help"])
    assert result.exit_code == 0


def test_serve_help():
    """serve --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0


def test_annotate_help():
    """annotate --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["annotate", "--help"])
    assert result.exit_code == 0


def test_extract_calibration_data_help():
    """extract-calibration-data --help exits cleanly"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["extract-calibration-data", "--help"])
    assert result.exit_code == 0


# ── Error handling tests ──

def test_run_without_trap_id_gives_error():
    """run without --trap-id, --trap-file, or --category exits with code 1"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1


def test_report_missing_file():
    """report with nonexistent file exits with code 1"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["report", "/nonexistent/file.json"])
    assert result.exit_code == 1


def test_batch_missing_file():
    """batch with nonexistent config exits with code 1"""
    from typer.testing import CliRunner

    from agent_trust_lab.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["batch", "/nonexistent/batch.yaml"])
    assert result.exit_code == 1
