"""Base harness for CLI-based agent adapters (OpenCode, Claude Code, Gemini CLI).

Extracts shared subprocess invocation, stub fallback, and step-type logic
into BaseCLIHarness so each concrete adapter is a thin override focused on
adapter_name, binary name, API key env var, and CLI prompt flag.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

logger = get_logger("adapters._cli_base")


# ---------------------------------------------------------------------------
# Module-level helpers (shared by all CLI harnesses)
# ---------------------------------------------------------------------------


def _find_cli_binary(name: str) -> Optional[str]:
    """Find the CLI binary on PATH, returning the path or None."""
    return shutil.which(name)


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


# ---------------------------------------------------------------------------
# BaseCLIHarness
# ---------------------------------------------------------------------------


@dataclass
class BaseCLIHarness(AgentHarness):
    """Shared base for CLI-driven agent harnesses.

    Subclasses only need to override:
      - adapter_name
      - _binary_name
      - _api_key_env
      - _cli_prompt_flag
      - _cli_display_name (optional, defaults to _binary_name)
      - from_config (classmethod)
    """

    # -- dataclass fields shared by all CLI harnesses -----------------------
    model: str = ""
    timeout: int = 300
    api_key: str = ""
    base_url: str = ""
    workspace: str = ""
    strict_mode: bool = False

    # -- properties subclasses MUST override --------------------------------

    @property
    def adapter_name(self) -> str:
        """Used in trajectory metadata and log messages. Subclasses MUST override."""
        raise NotImplementedError

    @property
    def _binary_name(self) -> str:
        """CLI binary name (e.g. 'opencode', 'claude', 'gemini')."""
        raise NotImplementedError

    @property
    def _api_key_env(self) -> str:
        """Environment variable name for the API key."""
        raise NotImplementedError

    @property
    def _cli_prompt_flag(self) -> str:
        """CLI flag for passing the prompt (e.g. '--print', '-p', '--prompt')."""
        raise NotImplementedError

    @property
    def _cli_display_name(self) -> str:
        """Human-readable CLI name for error/log messages. Defaults to _binary_name."""
        return self._binary_name

    # -- from_config -------------------------------------------------------

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "BaseCLIHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )

    # -- main entry point --------------------------------------------------

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
            logger.warning(
                "%s CLI call failed, falling back to stub: %s",
                self.__class__.__name__,
                e,
            )
            return self._run_stub(task, tools, max_steps, policy_rules, error=str(e))

    # -- CLI invocation (real) ---------------------------------------------

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

        steps.append(self._init_step())

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
                f"No API key configured for {self._cli_display_name} CLI. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file, "
                "or pass --api-key to the CLI."
            )

        binary = _find_cli_binary(self._binary_name)
        if not binary:
            raise FileNotFoundError(f"{self._binary_name} CLI not found on PATH")

        prompt = _build_task_with_tools(task, tools)
        cmd = [binary, self._cli_prompt_flag, prompt]
        env = os.environ.copy()
        env[self._api_key_env] = self.api_key

        logger.debug("Running %s: %s %s", self._binary_name, cmd[0], cmd[1])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{self._binary_name} CLI timed out after {self.timeout}s"
            )

        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(
                f"{self._binary_name} CLI exited with code {result.returncode}: "
                f"{result.stderr[:500]}"
            )

        if result.stdout:
            steps.append(
                TrajectoryStep(
                    type="cli_stdout",
                    content=result.stdout.strip(),
                    metadata={
                        "exit_code": result.returncode,
                        "adapter": self.adapter_name,
                    },
                )
            )
        if result.stderr:
            steps.append(
                TrajectoryStep(
                    type="cli_stderr",
                    content=result.stderr.strip(),
                    metadata={
                        "exit_code": result.returncode,
                        "adapter": self.adapter_name,
                    },
                )
            )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=result.stderr,
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "adapter": self.adapter_name,
                "model": self.model,
                "stub": False,
                "exit_code": result.returncode,
            },
        )

    # -- stub fallback -----------------------------------------------------

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

        steps.append(self._init_step())

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
        class_name = self.__class__.__name__
        steps.append(
            TrajectoryStep(
                type="cli_stdout",
                content=(
                    f"[{class_name}] Stub execution for task: {task[:200]}\n"
                    f"Available tools: {', '.join(tool_names)}\n"
                    + (f"Error: {error[:200]}" if error else "")
                ),
                metadata={"adapter": self.adapter_name, "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "adapter": self.adapter_name,
                "model": self.model,
                "stub": True,
            },
        )

    # -- hooks -------------------------------------------------------------

    def _init_step(self) -> TrajectoryStep:
        """Return the harness_init step inserted at the start of every trajectory."""
        return TrajectoryStep(
            type="harness_init",
            content=(
                f"{self.__class__.__name__} initialized: "
                f"{self._cli_display_name} CLI wrapper"
            ),
            metadata={"adapter": self.adapter_name, "model": self.model},
        )
