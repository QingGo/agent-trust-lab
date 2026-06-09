"""CLI-based agent harnesses — thin subclasses of BaseCLIHarness.

Each harness is a ~20 line override that only customises:
  - adapter_name      (used in trajectory metadata)
  - _binary_name      (CLI binary on PATH)
  - _api_key_env      (environment variable for the API key)
  - _cli_prompt_flag  (flag for passing the prompt, e.g. '--print')
  - _cli_display_name (optional human-readable name, defaults to _binary_name)
  - from_config       (classmethod)

All shared subprocess invocation, stub fallback, and step-type logic lives in
``agent_trust_lab.adapters._cli_base.BaseCLIHarness``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_trust_lab.adapters._cli_base import (
    BaseCLIHarness,
    _build_task_with_tools,
    _format_tools_as_text,
)
from agent_trust_lab.adapters.registry import register_adapter

if TYPE_CHECKING:
    from agent_trust_lab.config import EvaluationConfig


# ---------------------------------------------------------------------------
# OpenCodeHarness
# ---------------------------------------------------------------------------


@register_adapter("opencode")
@dataclass
class OpenCodeHarness(BaseCLIHarness):
    @property
    def adapter_name(self) -> str:
        return "opencode"

    @property
    def _binary_name(self) -> str:
        return "opencode"

    @property
    def _api_key_env(self) -> str:
        return "OPENCODE_API_KEY"

    @property
    def _cli_prompt_flag(self) -> str:
        return "--print"

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "OpenCodeHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )


# ---------------------------------------------------------------------------
# ClaudeCodeHarness
# ---------------------------------------------------------------------------


@register_adapter("claude-code")
@dataclass
class ClaudeCodeHarness(BaseCLIHarness):
    @property
    def adapter_name(self) -> str:
        return "claude-code"

    @property
    def _binary_name(self) -> str:
        return "claude"

    @property
    def _api_key_env(self) -> str:
        return "ANTHROPIC_API_KEY"

    @property
    def _cli_prompt_flag(self) -> str:
        return "-p"

    @property
    def _cli_display_name(self) -> str:
        return "Claude Code"

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "ClaudeCodeHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )


# ---------------------------------------------------------------------------
# GeminiCLIHarness
# ---------------------------------------------------------------------------


@register_adapter("gemini-cli")
@dataclass
class GeminiCLIHarness(BaseCLIHarness):
    @property
    def adapter_name(self) -> str:
        return "gemini-cli"

    @property
    def _binary_name(self) -> str:
        return "gemini"

    @property
    def _api_key_env(self) -> str:
        return "GEMINI_API_KEY"

    @property
    def _cli_prompt_flag(self) -> str:
        return "--prompt"

    @property
    def _cli_display_name(self) -> str:
        return "Gemini"

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "GeminiCLIHarness":
        return cls(
            model=config.model,
            timeout=config.timeout * 2,
            api_key=config.api_key,
        )
