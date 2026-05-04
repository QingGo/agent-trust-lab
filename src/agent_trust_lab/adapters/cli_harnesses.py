import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_trust_lab.adapters.registry import register_adapter
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.trajectory import (
    AgentHarness,
    SecureTrajectory,
    SecurityEvent,
    TrajectoryStep,
)
from agent_trust_lab.sandbox.filter import filter_command

if TYPE_CHECKING:
    from agent_trust_lab.config import EvaluationConfig

logger = get_logger("adapters.cli_harnesses")


def _format_tools_as_text(tools: List[Dict[str, Any]]) -> str:
    """Convert tool definitions to a text description for CLI agent context."""
    if not tools:
        return ""
    lines = ["Available tools:"]
    for t in tools:
        name = t.get("name", "unknown")
        desc = t.get("description", f"Execute {name} operation.")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _build_task_with_tools(task: str, tools: List[Dict[str, Any]]) -> str:
    """Augment the task with tool descriptions as text context."""
    tools_text = _format_tools_as_text(tools)
    if tools_text:
        return f"{task}\n\n{tools_text}"
    return task


def _find_cli_binary(name: str) -> Optional[str]:
    """Find the CLI binary on PATH, returning the path or None."""
    return shutil.which(name)


@register_adapter("opencode")
@dataclass
class OpenCodeHarness(AgentHarness):
    model: str = ""
    timeout: int = 300
    api_key: str = ""

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "OpenCodeHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        state_snapshot_paths: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        try:
            return self._run_with_cli(task, tools, max_steps, policy_rules)
        except Exception as e:
            logger.warning("OpenCodeHarness CLI call failed, falling back to stub: %s", e)
            return self._run_stub(task, tools, max_steps, policy_rules, error=str(e))

    def _run_with_cli(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="OpenCodeHarness initialized: opencode CLI wrapper",
                metadata={"adapter": "opencode", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        if not self.api_key:
            raise RuntimeError(
                "No API key configured for opencode CLI. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file, "
                "or pass --api-key to the CLI."
            )

        binary = _find_cli_binary("opencode")
        if not binary:
            raise FileNotFoundError("opencode CLI not found on PATH")

        prompt = _build_task_with_tools(task, tools)
        cmd = [binary, "--print", prompt]
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = self.api_key

        logger.debug("Running opencode: %s", cmd[:2])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"opencode CLI timed out after {self.timeout}s")

        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(
                f"opencode CLI exited with code {result.returncode}: {result.stderr[:500]}"
            )

        if result.stdout:
            steps.append(
                TrajectoryStep(
                    type="cli_stdout",
                    content=result.stdout.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "opencode"},
                )
            )
        if result.stderr:
            steps.append(
                TrajectoryStep(
                    type="cli_stderr",
                    content=result.stderr.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "opencode"},
                )
            )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=result.stderr,
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "adapter": "opencode",
                "model": self.model,
                "stub": False,
                "exit_code": result.returncode,
            },
        )

    def _run_stub(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        error: str = "",
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="OpenCodeHarness initialized: opencode CLI wrapper",
                metadata={"adapter": "opencode", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="cli_stdout",
                content=(
                    f"[OpenCodeHarness] Stub execution for task: {task[:200]}\n"
                    f"Available tools: {', '.join(tool_names)}\n"
                    + (f"Error: {error[:200]}" if error else "")
                ),
                metadata={"adapter": "opencode", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={"adapter": "opencode", "model": self.model, "stub": True},
        )


@register_adapter("claude-code")
@dataclass
class ClaudeCodeHarness(AgentHarness):
    model: str = ""
    timeout: int = 300
    api_key: str = ""

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "ClaudeCodeHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        state_snapshot_paths: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        try:
            return self._run_with_cli(task, tools, max_steps, policy_rules)
        except Exception as e:
            logger.warning("ClaudeCodeHarness CLI call failed, falling back to stub: %s", e)
            return self._run_stub(task, tools, max_steps, policy_rules, error=str(e))

    def _run_with_cli(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="ClaudeCodeHarness initialized: Claude Code CLI wrapper",
                metadata={"adapter": "claude-code", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        if not self.api_key:
            raise RuntimeError(
                "No API key configured for Claude Code CLI. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file, "
                "or pass --api-key to the CLI."
            )

        binary = _find_cli_binary("claude")
        if not binary:
            raise FileNotFoundError("claude CLI not found on PATH")

        prompt = _build_task_with_tools(task, tools)
        cmd = [binary, "-p", prompt]
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = self.api_key

        logger.debug("Running claude: %s %s", cmd[0], cmd[1])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")

        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(
                f"claude CLI exited with code {result.returncode}: {result.stderr[:500]}"
            )

        if result.stdout:
            steps.append(
                TrajectoryStep(
                    type="cli_stdout",
                    content=result.stdout.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "claude-code"},
                )
            )
        if result.stderr:
            steps.append(
                TrajectoryStep(
                    type="cli_stderr",
                    content=result.stderr.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "claude-code"},
                )
            )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=result.stderr,
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "adapter": "claude-code",
                "model": self.model,
                "stub": False,
                "exit_code": result.returncode,
            },
        )

    def _run_stub(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        error: str = "",
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="ClaudeCodeHarness initialized: Claude Code CLI wrapper",
                metadata={"adapter": "claude-code", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="cli_stdout",
                content=(
                    f"[ClaudeCodeHarness] Stub execution for task: {task[:200]}\n"
                    f"Available tools: {', '.join(tool_names)}\n"
                    + (f"Error: {error[:200]}" if error else "")
                ),
                metadata={"adapter": "claude-code", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={"adapter": "claude-code", "model": self.model, "stub": True},
        )


@register_adapter("gemini-cli")
@dataclass
class GeminiCLIHarness(AgentHarness):
    model: str = ""
    timeout: int = 300
    api_key: str = ""

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "GeminiCLIHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        state_snapshot_paths: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        try:
            return self._run_with_cli(task, tools, max_steps, policy_rules)
        except Exception as e:
            logger.warning("GeminiCLIHarness CLI call failed, falling back to stub: %s", e)
            return self._run_stub(task, tools, max_steps, policy_rules, error=str(e))

    def _run_with_cli(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="GeminiCLIHarness initialized: Gemini CLI wrapper",
                metadata={"adapter": "gemini-cli", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        if not self.api_key:
            raise RuntimeError(
                "No API key configured for Gemini CLI. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file, "
                "or pass --api-key to the CLI."
            )

        binary = _find_cli_binary("gemini")
        if not binary:
            raise FileNotFoundError("gemini CLI not found on PATH")

        prompt = _build_task_with_tools(task, tools)
        cmd = [binary, "--prompt", prompt]
        env = os.environ.copy()
        env["GEMINI_API_KEY"] = self.api_key

        logger.debug("Running gemini: %s %s", cmd[0], cmd[1])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gemini CLI timed out after {self.timeout}s")

        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(
                f"gemini CLI exited with code {result.returncode}: {result.stderr[:500]}"
            )

        if result.stdout:
            steps.append(
                TrajectoryStep(
                    type="cli_stdout",
                    content=result.stdout.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "gemini-cli"},
                )
            )
        if result.stderr:
            steps.append(
                TrajectoryStep(
                    type="cli_stderr",
                    content=result.stderr.strip(),
                    metadata={"exit_code": result.returncode, "adapter": "gemini-cli"},
                )
            )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=result.stderr,
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "adapter": "gemini-cli",
                "model": self.model,
                "stub": False,
                "exit_code": result.returncode,
            },
        )

    def _run_stub(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
        error: str = "",
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_rules_applied: List[str] = list(policy_rules) if policy_rules else []
        actual_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content="GeminiCLIHarness initialized: Gemini CLI wrapper",
                metadata={"adapter": "gemini-cli", "model": self.model},
            )
        )

        filtered = filter_command(task)
        if not filtered.allowed:
            security_events.append(
                SecurityEvent(
                    event_type="cmd_filtered",
                    description=filtered.description,
                    step_index=0,
                )
            )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="cli_stdout",
                content=(
                    f"[GeminiCLIHarness] Stub execution for task: {task[:200]}\n"
                    f"Available tools: {', '.join(tool_names)}\n"
                    + (f"Error: {error[:200]}" if error else "")
                ),
                metadata={"adapter": "gemini-cli", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={"adapter": "gemini-cli", "model": self.model, "stub": True},
        )
