from unittest.mock import patch

from agent_trust_lab.adapters import CodexHarness, LangChainHarness, OpenAIFunctionHarness
from agent_trust_lab.models.trajectory import AgentHarness, SecureTrajectory


class TestLangChainHarness:
    def test_default_construction(self):
        harness = LangChainHarness()
        assert harness.model == "deepseek-v4-flash"
        assert harness.temperature == 0.0

    def test_custom_construction(self):
        harness = LangChainHarness(model="gpt-4", temperature=0.5)
        assert harness.model == "gpt-4"
        assert harness.temperature == 0.5

    def test_run_returns_secure_trajectory(self):
        harness = LangChainHarness()
        trajectory = harness.run(task="Test task", tools=[{"name": "file_read"}])
        assert len(trajectory.steps) >= 1
        assert trajectory.metadata["adapter"] == "langchain"
        assert "stub" in trajectory.metadata

    def test_stub_fallback_when_no_api_key(self):
        harness = LangChainHarness(api_key="none", base_url="https://invalid.url")
        trajectory = harness.run(task="Test task", tools=[{"name": "file_read"}])
        assert trajectory.metadata["stub"] is True
        assert len(trajectory.steps) >= 3

    def test_run_steps_have_expected_types(self):
        harness = LangChainHarness()
        trajectory = harness.run(task="test", tools=[{"name": "search"}])
        step_types = [s.type for s in trajectory.steps]
        assert "harness_init" in step_types
        assert any(t in step_types for t in ("thought", "action", "observation"))

    def test_run_with_dangerous_command(self):
        harness = LangChainHarness()
        trajectory = harness.run(task="sudo rm -rf /", tools=[{"name": "shell"}])
        assert len(trajectory.security_events) >= 1
        assert trajectory.security_events[0].event_type == "cmd_filtered"

    def test_run_with_policy_rules(self):
        harness = LangChainHarness()
        trajectory = harness.run(task="test", tools=[], policy_rules=["Unauthorized access"])
        assert "Unauthorized access" in trajectory.policy_rules_applied

    def test_is_agent_harness_subclass(self):
        assert isinstance(LangChainHarness(), AgentHarness)


class TestOpenAIFunctionHarness:
    def test_default_construction(self):
        harness = OpenAIFunctionHarness()
        assert harness.model == "gpt-4o-mini"

    def test_run_returns_secure_trajectory(self):
        harness = OpenAIFunctionHarness()
        trajectory = harness.run(task="Test", tools=[{"name": "calc"}])
        assert len(trajectory.steps) >= 3
        assert trajectory.metadata["adapter"] == "openai-functions"
        assert trajectory.metadata["stub"] is True

    def test_run_steps_have_expected_types(self):
        harness = OpenAIFunctionHarness()
        trajectory = harness.run(task="test", tools=[{"name": "search"}])
        step_types = [s.type for s in trajectory.steps]
        assert "function_call" in step_types
        assert "function_result" in step_types

    def test_is_agent_harness_subclass(self):
        assert isinstance(OpenAIFunctionHarness(), AgentHarness)


class TestCodexHarness:
    def test_default_construction(self):
        harness = CodexHarness()
        assert harness.model == "gpt-4o-mini"
        assert harness.codebase_path is None

    def test_custom_construction(self):
        harness = CodexHarness(codebase_path="/path/to/repo", test_command="pytest")
        assert harness.codebase_path == "/path/to/repo"
        assert harness.test_command == "pytest"

    def test_run_returns_secure_trajectory(self):
        harness = CodexHarness()
        trajectory = harness.run(task="Fix bug", tools=[{"name": "file_read"}])
        assert len(trajectory.steps) >= 3
        assert trajectory.metadata["adapter"] == "codex"
        assert trajectory.metadata["stub"] is True

    def test_run_steps_have_expected_types(self):
        harness = CodexHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        step_types = [s.type for s in trajectory.steps]
        assert "code_action" in step_types
        assert "code_result" in step_types

    def test_run_with_dangerous_command(self):
        harness = CodexHarness()
        trajectory = harness.run(task="eval malicious", tools=[{"name": "shell"}])
        assert len(trajectory.security_events) >= 1

    def test_is_agent_harness_subclass(self):
        assert isinstance(CodexHarness(), AgentHarness)

    def test_new_fields_exist(self):
        harness = CodexHarness()
        assert harness.api_key == ""
        assert harness.base_url == ""
        assert harness.thinking_enabled is False
        assert harness.reasoning_effort == ""
        assert harness.temperature == 0.0
        assert harness.timeout == 120

    def test_from_config_passes_all_fields(self):
        from agent_trust_lab.config import EvaluationConfig

        config = EvaluationConfig(
            model="deepseek-code",
            codebase_path="/test/project",
            api_key="sk-test",
            base_url="https://api.example.com",
            thinking_enabled=True,
            reasoning_effort="high",
        )
        h = CodexHarness.from_config(config)
        assert h.model == "deepseek-code"
        assert h.codebase_path == "/test/project"
        assert h.api_key == "sk-test"
        assert h.base_url == "https://api.example.com"
        assert h.thinking_enabled is True
        assert h.reasoning_effort == "high"

    def test_run_has_code_thought_types(self):
        harness = CodexHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        step_types = [s.type for s in trajectory.steps]
        assert "code_thought" in step_types

    def test_harness_init_includes_codebase(self):
        harness = CodexHarness(codebase_path="/home/dev/repo")
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        init_step = trajectory.steps[0]
        assert "/home/dev/repo" in init_step.content

    def test_run_metadata_has_codex_adapter(self):
        harness = CodexHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        assert trajectory.metadata["adapter"] == "codex"

    @patch("agent_trust_lab.adapters.harnesses.CodexHarness._run_with_llm")
    def test_run_tries_real_llm_first(self, mock_llm):
        from agent_trust_lab.models.trajectory import TrajectoryStep

        harness = CodexHarness(model="test-model")
        mock_llm.return_value = SecureTrajectory(
            steps=[TrajectoryStep(type="code_thought", content="Hello")],
            security_events=[],
            metadata={"adapter": "codex", "model": "test-model", "stub": False},
        )
        trajectory = harness.run(task="test", tools=[{"name": "file_read"}])
        assert mock_llm.called
        assert trajectory.metadata["stub"] is False

    @patch("agent_trust_lab.adapters.harnesses.CodexHarness._run_with_llm")
    def test_run_falls_back_to_stub_on_error(self, mock_llm):
        harness = CodexHarness()
        mock_llm.side_effect = RuntimeError("API failure")
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        assert trajectory.metadata["stub"] is True
        assert any("API failure" in s.content for s in trajectory.steps)


class TestToolWhitelist:
    def test_langchain_unauthorized_tool_generates_event(self):
        from unittest.mock import MagicMock, patch

        harness = LangChainHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "suspicious_delete"
        mock_tc.function.arguments = '{"path": "/tmp/x"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "Let me run this tool"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "file_read"}],
                )

        unauthorized = [
            e for e in trajectory.security_events if e.event_type == "unauthorized_tool"
        ]
        assert len(unauthorized) >= 1
        assert "suspicious_delete" in unauthorized[0].description

    def test_langchain_authorized_tool_no_event(self):
        from unittest.mock import MagicMock, patch

        harness = LangChainHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "file_read"
        mock_tc.function.arguments = '{"path": "/tmp/x"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "Let me read"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "file_read"}],
                )

        unauthorized = [
            e for e in trajectory.security_events if e.event_type == "unauthorized_tool"
        ]
        assert len(unauthorized) == 0

    def test_codex_unauthorized_tool_generates_event(self):
        from unittest.mock import MagicMock, patch

        harness = CodexHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "sudo_exec"  # NOT in authorized tools
        mock_tc.function.arguments = '{"cmd": "ls"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "Running"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "code_search"}],
                )

        unauthorized = [
            e for e in trajectory.security_events if e.event_type == "unauthorized_tool"
        ]
        assert len(unauthorized) >= 1
        assert "sudo_exec" in unauthorized[0].description


class TestToolArgumentFilter:
    def test_langchain_blocks_dangerous_argument(self):
        from unittest.mock import MagicMock, patch

        harness = LangChainHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "shell"
        mock_tc.function.arguments = "sudo rm -rf /"

        mock_choice = MagicMock()
        mock_choice.message.content = "Executing"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "shell"}],
                )

        arg_filtered = [
            e
            for e in trajectory.security_events
            if e.event_type == "cmd_filtered" and "Tool argument blocked" in e.description
        ]
        assert len(arg_filtered) >= 1

    def test_langchain_safe_argument_no_event(self):
        from unittest.mock import MagicMock, patch

        harness = LangChainHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "shell"
        mock_tc.function.arguments = '{"cmd": "ls -la"}'

        mock_choice = MagicMock()
        mock_choice.message.content = "Listing"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "shell"}],
                )

        arg_filtered = [
            e
            for e in trajectory.security_events
            if e.event_type == "cmd_filtered" and "Tool argument blocked" in e.description
        ]
        assert len(arg_filtered) == 0

    def test_codex_blocks_dangerous_argument(self):
        from unittest.mock import MagicMock, patch

        harness = CodexHarness()

        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "shell"
        mock_tc.function.arguments = "eval malicious_code"

        mock_choice = MagicMock()
        mock_choice.message.content = "Running code"
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent_trust_lab.llm.create_openai_client", return_value=mock_client):
            with patch("agent_trust_lab.llm.get_api_key", return_value="sk-test"):
                trajectory = harness.run(
                    task="test",
                    tools=[{"name": "shell"}],
                )

        arg_filtered = [
            e
            for e in trajectory.security_events
            if e.event_type == "cmd_filtered" and "Tool argument blocked" in e.description
        ]
        assert len(arg_filtered) >= 1


class TestOpenCodeHarness:
    def test_default_construction(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        assert harness.model == ""
        assert harness.timeout == 300

    def test_custom_construction(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness(model="gpt-4", timeout=600, api_key="sk-test")
        assert harness.model == "gpt-4"
        assert harness.timeout == 600
        assert harness.api_key == "sk-test"

    def test_run_returns_secure_trajectory(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="Test task", tools=[{"name": "file_read"}])
        assert len(trajectory.steps) >= 1
        assert trajectory.metadata["adapter"] == "opencode"
        assert trajectory.metadata["stub"] is True

    def test_run_steps_have_expected_types(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        step_types = [s.type for s in trajectory.steps]
        assert "harness_init" in step_types
        assert "cli_stdout" in step_types

    def test_run_with_dangerous_command(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="sudo rm -rf /", tools=[{"name": "shell"}])
        assert len(trajectory.security_events) >= 1
        assert trajectory.security_events[0].event_type == "cmd_filtered"

    def test_run_with_policy_rules(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="test", tools=[], policy_rules=["No file access"])
        assert "No file access" in trajectory.policy_rules_applied

    def test_is_agent_harness_subclass(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        assert isinstance(OpenCodeHarness(), AgentHarness)

    def test_from_config_passes_all_fields(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness
        from agent_trust_lab.config import EvaluationConfig

        config = EvaluationConfig(
            model="deepseek-v4-flash",
            api_key="sk-test",
            timeout=60,
        )
        h = OpenCodeHarness.from_config(config)
        assert h.model == "deepseek-v4-flash"
        assert h.api_key == "sk-test"
        assert h.timeout == 120

    def test_run_stub_includes_tools(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(
            task="Fix bug",
            tools=[{"name": "file_read"}, {"name": "shell"}],
        )
        stdout_step = [s for s in trajectory.steps if s.type == "cli_stdout"][0]
        assert "file_read" in stdout_step.content
        assert "shell" in stdout_step.content
        assert "Fix bug" in stdout_step.content

    def test_stub_error_included_in_content(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness._run_stub(task="test", tools=[], error="CLI not found")
        stdout_step = [s for s in trajectory.steps if s.type == "cli_stdout"][0]
        assert "CLI not found" in stdout_step.content

    def test_harness_init_includes_adapter_name(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="test", tools=[])
        init_step = trajectory.steps[0]
        assert init_step.type == "harness_init"
        assert "opencode" in init_step.metadata.get("adapter", "")


class TestClaudeCodeHarness:
    def test_default_construction(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        assert harness.model == ""
        assert harness.timeout == 300

    def test_run_returns_secure_trajectory(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        trajectory = harness.run(task="Test task", tools=[{"name": "file_read"}])
        assert len(trajectory.steps) >= 1
        assert trajectory.metadata["adapter"] == "claude-code"
        assert trajectory.metadata["stub"] is True

    def test_run_steps_have_expected_types(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        step_types = [s.type for s in trajectory.steps]
        assert "harness_init" in step_types
        assert "cli_stdout" in step_types

    def test_is_agent_harness_subclass(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        assert isinstance(ClaudeCodeHarness(), AgentHarness)

    def test_from_config_passes_all_fields(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness
        from agent_trust_lab.config import EvaluationConfig

        config = EvaluationConfig(
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            timeout=60,
        )
        h = ClaudeCodeHarness.from_config(config)
        assert h.model == "claude-sonnet-4-20250514"
        assert h.api_key == "sk-ant-test"
        assert h.timeout == 120

    def test_run_stub_includes_adapter_name(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        trajectory = harness.run(task="test", tools=[{"name": "grep"}])
        stdout_step = [s for s in trajectory.steps if s.type == "cli_stdout"][0]
        assert "ClaudeCodeHarness" in stdout_step.content or "claude" in stdout_step.content.lower()


class TestGeminiCLIHarness:
    def test_default_construction(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        assert harness.model == ""
        assert harness.timeout == 300

    def test_run_returns_secure_trajectory(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="Test task", tools=[{"name": "file_read"}])
        assert len(trajectory.steps) >= 1
        assert trajectory.metadata["adapter"] == "gemini-cli"
        assert trajectory.metadata["stub"] is True

    def test_run_steps_have_expected_types(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="test", tools=[{"name": "shell"}])
        step_types = [s.type for s in trajectory.steps]
        assert "harness_init" in step_types
        assert "cli_stdout" in step_types

    def test_is_agent_harness_subclass(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        assert isinstance(GeminiCLIHarness(), AgentHarness)

    def test_from_config_passes_all_fields(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness
        from agent_trust_lab.config import EvaluationConfig

        config = EvaluationConfig(
            model="gemini-2.5-pro",
            api_key="sk-gemini-test",
            timeout=60,
        )
        h = GeminiCLIHarness.from_config(config)
        assert h.model == "gemini-2.5-pro"
        assert h.api_key == "sk-gemini-test"
        assert h.timeout == 120

    def test_run_with_dangerous_command(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="eval(malicious)", tools=[{"name": "shell"}])
        assert len(trajectory.security_events) >= 1
        assert trajectory.security_events[0].event_type == "cmd_filtered"

    def test_harness_init_includes_adapter_name(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="test", tools=[])
        init_step = trajectory.steps[0]
        assert init_step.type == "harness_init"
        assert "gemini-cli" in init_step.metadata.get("adapter", "")


class TestCLIHarnessRealCLI:
    def test_opencode_real_cli_fallback_to_stub(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="test", tools=[{"name": "file_read"}])
        assert trajectory.metadata["stub"] is True

    def test_claude_code_real_cli_fallback_to_stub(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        trajectory = harness.run(task="test", tools=[{"name": "file_read"}])
        assert trajectory.metadata["stub"] is True

    def test_gemini_real_cli_fallback_to_stub(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="test", tools=[{"name": "file_read"}])
        assert trajectory.metadata["stub"] is True

    def test_opencode_cli_not_found_handled_gracefully(self):
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        harness = OpenCodeHarness()
        trajectory = harness.run(task="test", tools=[])
        assert trajectory.metadata["stub"] is True

    def test_claude_code_cli_not_found_handled_gracefully(self):
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        harness = ClaudeCodeHarness()
        trajectory = harness.run(task="test", tools=[])
        assert trajectory.metadata["stub"] is True

    def test_gemini_cli_not_found_handled_gracefully(self):
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        harness = GeminiCLIHarness()
        trajectory = harness.run(task="test", tools=[])
        assert trajectory.metadata["stub"] is True


class TestFormatToolsAsText:
    def test_empty_tools(self):
        from agent_trust_lab.adapters.cli_harnesses import _format_tools_as_text

        assert _format_tools_as_text([]) == ""

    def test_single_tool(self):
        from agent_trust_lab.adapters.cli_harnesses import _format_tools_as_text

        result = _format_tools_as_text([{"name": "file_read"}])
        assert "file_read" in result
        assert "Available tools:" in result

    def test_multiple_tools(self):
        from agent_trust_lab.adapters.cli_harnesses import _format_tools_as_text

        result = _format_tools_as_text(
            [
                {"name": "file_read", "description": "Read a file"},
                {"name": "shell", "description": "Run shell commands"},
            ]
        )
        assert "file_read" in result
        assert "shell" in result
        assert "Read a file" in result
        assert "Run shell commands" in result

    def test_tool_without_description(self):
        from agent_trust_lab.adapters.cli_harnesses import _format_tools_as_text

        result = _format_tools_as_text([{"name": "search"}])
        assert "search" in result
        assert "Execute search operation" in result

    def test_tool_without_name(self):
        from agent_trust_lab.adapters.cli_harnesses import _format_tools_as_text

        result = _format_tools_as_text([{"description": "Does something"}])
        assert "unknown" in result


class TestBuildTaskWithTools:
    def test_no_tools_returns_task(self):
        from agent_trust_lab.adapters.cli_harnesses import _build_task_with_tools

        result = _build_task_with_tools("Do X", [])
        assert result == "Do X"

    def test_with_tools_appends_context(self):
        from agent_trust_lab.adapters.cli_harnesses import _build_task_with_tools

        result = _build_task_with_tools("Do X", [{"name": "grep", "description": "Search files"}])
        assert result.startswith("Do X")
        assert "grep" in result
        assert "Search files" in result
