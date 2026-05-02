"""Tests for adapter registry."""

from dataclasses import dataclass

import pytest

from agent_trust_lab.adapters.registry import (
    get_adapter_class,
    list_adapters,
    register_adapter,
    resolve,
)
from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.models.trajectory import AgentHarness, SecureTrajectory


class TestRegisterAdapter:
    def test_register_simple_class(self):
        @register_adapter("test_foo")
        @dataclass
        class TestFoo(AgentHarness):
            x: int = 1

            def run(self, task, tools, **kwargs):
                return SecureTrajectory(steps=[], security_events=[])

        assert "test_foo" in list_adapters()
        cls = resolve("test_foo")
        assert cls is TestFoo

    def test_register_two_classes(self):
        @register_adapter("test_alpha")
        @dataclass
        class Alpha(AgentHarness):
            def run(self, task, tools, **kwargs):
                return SecureTrajectory(steps=[], security_events=[])

        @register_adapter("test_beta")
        @dataclass
        class Beta(AgentHarness):
            def run(self, task, tools, **kwargs):
                return SecureTrajectory(steps=[], security_events=[])

        assert "test_alpha" in list_adapters()
        assert "test_beta" in list_adapters()
        assert resolve("test_alpha") is Alpha
        assert resolve("test_beta") is Beta

    def test_register_replaces_existing(self, caplog):
        @register_adapter("test_dup")
        @dataclass
        class First(AgentHarness):
            def run(self, task, tools, **kwargs):
                return SecureTrajectory(steps=[], security_events=[])

        @register_adapter("test_dup")
        @dataclass
        class Second(AgentHarness):
            def run(self, task, tools, **kwargs):
                return SecureTrajectory(steps=[], security_events=[])

        assert resolve("test_dup") is Second
        assert "re-registered" in caplog.text

    def test_unregistered_name_returns_none(self):
        assert resolve("nonexistent") is None
        assert get_adapter_class("nonexistent") is None


class TestListAdapters:
    def test_core_adapters_registered(self):
        adapters = list_adapters()
        assert "claude-code" in adapters
        assert "codex" in adapters
        assert "docker" in adapters
        assert "dry-run" in adapters
        assert "gemini-cli" in adapters
        assert "langchain" in adapters
        assert "openai" in adapters
        assert "opencode" in adapters

    def test_list_returns_sorted(self):
        adapters = list_adapters()
        assert adapters == sorted(adapters)


class TestFromConfig:
    def test_langchain_from_config(self):
        from agent_trust_lab.adapters.harnesses import LangChainHarness

        config = EvaluationConfig(
            model="test-model",
            thinking_enabled=True,
            reasoning_effort="max",
            base_url="https://example.com",
        )
        h = LangChainHarness.from_config(config)
        assert isinstance(h, LangChainHarness)
        assert h.model == "test-model"
        assert h.thinking_enabled is True
        assert h.reasoning_effort == "max"
        assert h.base_url == "https://example.com"

    def test_langchain_from_config_defaults(self):
        from agent_trust_lab.adapters.harnesses import LangChainHarness

        config = EvaluationConfig(model="deepseek-v4-flash")
        h = LangChainHarness.from_config(config)
        assert h.model == "deepseek-v4-flash"
        assert h.thinking_enabled is False
        assert h.reasoning_effort == ""

    def test_openai_from_config(self):
        from agent_trust_lab.adapters.harnesses import OpenAIFunctionHarness

        config = EvaluationConfig(model="gpt-4o-mini")
        h = OpenAIFunctionHarness.from_config(config)
        assert isinstance(h, OpenAIFunctionHarness)
        assert h.model == "gpt-4o-mini"

    def test_codex_from_config(self):
        from agent_trust_lab.adapters.harnesses import CodexHarness

        config = EvaluationConfig(model="deepseek-v4-flash", codebase_path="/path/to/code")
        h = CodexHarness.from_config(config)
        assert isinstance(h, CodexHarness)
        assert h.model == "deepseek-v4-flash"
        assert h.codebase_path == "/path/to/code"

    def test_codex_from_config_no_codebase(self):
        from agent_trust_lab.adapters.harnesses import CodexHarness

        config = EvaluationConfig(model="deepseek-v4-flash")
        h = CodexHarness.from_config(config)
        assert h.codebase_path is None

    def test_docker_from_config(self):
        from agent_trust_lab.sandbox.backends import DockerSandbox

        config = EvaluationConfig(
            sandbox_image="alpine:latest",
            timeout=60,
            sandbox_network=True,
            sandbox_tmpfs_size="128m",
            docker_host="unix:///var/run/podman.sock",
        )
        h = DockerSandbox.from_config(config)
        assert isinstance(h, DockerSandbox)
        assert h.image == "alpine:latest"
        assert h.timeout == 60
        assert h.network_enabled is True
        assert h.tmpfs_size == "128m"
        assert h.docker_host == "unix:///var/run/podman.sock"

    def test_docker_from_config_default_image(self):
        from agent_trust_lab.sandbox.backends import DockerSandbox

        config = EvaluationConfig(sandbox_image="")
        h = DockerSandbox.from_config(config)
        assert h.image == "docker.m.daocloud.io/library/busybox:latest"

    def test_dryrun_from_config(self):
        from agent_trust_lab.sandbox.backends import DryRunSandbox

        config = EvaluationConfig()
        h = DryRunSandbox.from_config(config)
        assert isinstance(h, DryRunSandbox)


class TestRegistryIntegration:
    @pytest.fixture(autouse=True)
    def _stub_hallukg(self):
        from unittest.mock import patch

        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            yield

    def test_resolve_langchain_harness(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.adapters.harnesses import LangChainHarness

        assert isinstance(harness, LangChainHarness)

    def test_resolve_codex_harness(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "code"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_code_control
version: "1.0.0"
severity: none
difficulty: trivial
category: code_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="codex",
            model="deepseek-v4-flash",
            codebase_path="/test/code",
            sandbox="dry-run",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.adapters.harnesses import CodexHarness

        assert isinstance(harness, CodexHarness)

    def test_resolve_sandbox_when_agent_empty(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="",
            sandbox="docker",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.sandbox.backends import DockerSandbox

        assert isinstance(harness, DockerSandbox)

    def test_resolve_unknown_raises(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps"
        trap_dir.mkdir()

        config = EvaluationConfig(
            agent_type="unknown_type",
            sandbox="unknown_sandbox",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        with pytest.raises(ValueError, match="Unknown harness"):
            orch.resolve_harness()

    def test_resolve_opencode_harness(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="opencode",
            model="deepseek-v4-flash",
            api_key="sk-test-opencode",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.adapters.cli_harnesses import OpenCodeHarness

        assert isinstance(harness, OpenCodeHarness)
        assert harness.api_key == "sk-test-opencode"

    def test_resolve_claude_code_harness(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "code"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_code_control
version: "1.0.0"
severity: none
difficulty: trivial
category: code_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="claude-code",
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-test",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.adapters.cli_harnesses import ClaudeCodeHarness

        assert isinstance(harness, ClaudeCodeHarness)
        assert harness.api_key == "sk-ant-test"

    def test_resolve_gemini_cli_harness(self, tmp_path):
        from agent_trust_lab.orchestrator import Orchestrator

        trap_dir = tmp_path / "traps" / "code"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_code_control
version: "1.0.0"
severity: none
difficulty: trivial
category: code_agent
base_task: "Test."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        config = EvaluationConfig(
            agent_type="gemini-cli",
            model="gemini-2.5-pro",
            api_key="sk-gemini-test",
            trap_library_path=str(trap_dir),
        )
        orch = Orchestrator(config)
        harness = orch.resolve_harness()
        from agent_trust_lab.adapters.cli_harnesses import GeminiCLIHarness

        assert isinstance(harness, GeminiCLIHarness)
        assert harness.api_key == "sk-gemini-test"
