from unittest.mock import patch


class TestWebUIBasics:
    def test_create_ui_returns_blocks(self):
        from gradio import Blocks

        from agent_trust_lab.web import create_ui

        demo = create_ui()
        assert isinstance(demo, Blocks)

    def test_get_trap_info_valid(self):
        from agent_trust_lab.web._shared import _get_trap_info

        info = _get_trap_info("parameter_hallucination_01")
        assert info is not None
        assert info["trap_id"] == "parameter_hallucination_01"
        assert info["trap_type"] == "parameter_hallucination"
        assert "base_task" in info
        assert "tools" in info

    def test_get_trap_info_missing(self):
        from agent_trust_lab.web._shared import _get_trap_info

        info = _get_trap_info("nonexistent_trap_999")
        assert info is None

    def test_build_trap_choices_has_categories(self):
        from agent_trust_lab.web._shared import _build_trap_choices

        choices = _build_trap_choices()
        assert isinstance(choices, dict)
        assert len(choices) >= 2
        assert "general_agent" in choices
        assert "code_agent" in choices

    def test_run_evaluation_returns_dict(self):
        from agent_trust_lab.web._shared import _run_evaluation

        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            result = _run_evaluation(
                trap_id="parameter_hallucination_01",
                model="deepseek-v4-flash",
                agent_type="langchain",
                sandbox="dry-run",
                thinking=False,
                effort="",
                mutate=False,
            )
        assert isinstance(result, dict)
        assert "trap_id" in result
        assert "steps_count" in result

    def test_run_evaluation_with_thinking(self):
        from agent_trust_lab.web._shared import _run_evaluation

        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            result = _run_evaluation(
                trap_id="benign_control_01",
                model="deepseek-v4-flash",
                agent_type="langchain",
                sandbox="dry-run",
                thinking=True,
                effort="high",
                mutate=False,
            )
        assert isinstance(result, dict)
        assert result["trap_id"] == "benign_control_01"


class TestWebCLI:
    def test_serve_help(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Launch the Gradio web UI" in result.stdout

    def test_serve_with_custom_host_port(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        with patch("agent_trust_lab.web.launch_ui") as mock_launch:
            result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8080"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(
            server_name="0.0.0.0", server_port=8080, share=False
        )
