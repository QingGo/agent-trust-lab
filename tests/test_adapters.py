from agent_trust_lab.adapters import CodexHarness, LangChainHarness, OpenAIFunctionHarness
from agent_trust_lab.models.trajectory import AgentHarness


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
        trajectory = harness.run(
            task="test", tools=[], policy_rules=["Unauthorized access"]
        )
        assert "Unauthorized access" in trajectory.policy_violations

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
