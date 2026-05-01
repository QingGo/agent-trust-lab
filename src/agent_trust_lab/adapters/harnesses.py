from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_trust_lab.models.trajectory import (
    AgentHarness,
    SecureTrajectory,
    SecurityEvent,
    TrajectoryStep,
)
from agent_trust_lab.sandbox.filter import filter_command


@dataclass
class LangChainHarness(AgentHarness):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_violations: List[str] = list(policy_rules) if policy_rules else []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content=f"LangChainHarness initialized: model={self.model}",
                metadata={"adapter": "langchain", "model": self.model},
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

        steps.append(
            TrajectoryStep(
                type="thought",
                content=f"[LangChain] Analyzing task: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="action",
                content=f"[LangChain] Would invoke tools: {', '.join(tool_names)}",
                tools_called=tool_names,
                metadata={"adapter": "langchain", "status": "stub"},
            )
        )

        steps.append(
            TrajectoryStep(
                type="observation",
                content=(
                    "[LangChainHarness] Stub execution: agent reasoning loop "
                    "would run here. No actual LLM calls performed."
                ),
                metadata={"adapter": "langchain", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_violations=policy_violations,
            metadata={"adapter": "langchain", "model": self.model, "stub": True},
        )


@dataclass
class OpenAIFunctionHarness(AgentHarness):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_violations: List[str] = list(policy_rules) if policy_rules else []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content=f"OpenAIFunctionHarness initialized: model={self.model}",
                metadata={"adapter": "openai-functions", "model": self.model},
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

        steps.append(
            TrajectoryStep(
                type="thought",
                content=f"[OpenAI Functions] Planning for task: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="function_call",
                content=f"[OpenAI Functions] Would call: {', '.join(tool_names)}",
                tools_called=tool_names,
                metadata={"adapter": "openai-functions", "status": "stub"},
            )
        )

        steps.append(
            TrajectoryStep(
                type="function_result",
                content=(
                    "[OpenAIFunctionHarness] Stub execution: function calling "
                    "loop would run here. No actual API calls performed."
                ),
                metadata={"adapter": "openai-functions", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_violations=policy_violations,
            metadata={"adapter": "openai-functions", "model": self.model, "stub": True},
        )


@dataclass
class CodexHarness(AgentHarness):
    model: str = "gpt-4o-mini"
    codebase_path: Optional[str] = None
    test_command: Optional[str] = None

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_violations: List[str] = list(policy_rules) if policy_rules else []

        steps.append(
            TrajectoryStep(
                type="harness_init",
                content=(
                    f"CodexHarness initialized: model={self.model}, "
                    f"codebase={self.codebase_path or 'none'}"
                ),
                metadata={
                    "adapter": "codex",
                    "model": self.model,
                    "codebase_path": self.codebase_path,
                },
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

        steps.append(
            TrajectoryStep(
                type="thought",
                content=f"[Codex] Exploring codebase for task: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        tool_names = [t.get("name", "unknown") for t in tools]
        steps.append(
            TrajectoryStep(
                type="code_action",
                content=f"[Codex] Would use tools: {', '.join(tool_names)}",
                tools_called=tool_names,
                metadata={"adapter": "codex", "status": "stub"},
            )
        )

        steps.append(
            TrajectoryStep(
                type="code_result",
                content=(
                    "[CodexHarness] Stub execution: code generation/execution "
                    "loop would run here. No actual code operations performed."
                ),
                metadata={"adapter": "codex", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_violations=policy_violations,
            metadata={"adapter": "codex", "model": self.model, "stub": True},
        )
