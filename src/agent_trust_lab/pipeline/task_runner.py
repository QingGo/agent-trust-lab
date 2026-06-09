"""Task runner for orchestrator pipeline.

Handles harness resolution, template rendering, trap injection,
harness execution with error handling, and tool call assertions.
"""

from typing import Dict, List, Optional

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.trajectory import (
    AgentHarness,
    SecureTrajectory,
    SecurityEvent,
    TrajectoryStep,
)
from agent_trust_lab.models.trap import EnhancedTrapDef
from agent_trust_lab.traps.manager import TrapManager

logger = get_logger("orchestrator")


class TaskRunner:
    """Handles trap → task execution flow."""

    def __init__(self, config: EvaluationConfig, trap_manager: TrapManager) -> None:
        self._config = config
        self._trap_manager = trap_manager

    def resolve_harness(self, agent_type: Optional[str] = None) -> AgentHarness:
        import agent_trust_lab.adapters  # noqa: F401 ensure registry is populated
        from agent_trust_lab.adapters.registry import list_adapters, resolve

        agent = agent_type or self._config.agent_type
        sandbox_type = self._config.sandbox.lower()

        for name in (agent, sandbox_type):
            cls = resolve(name)
            if cls is not None:
                from_config = getattr(cls, "from_config", None)
                if from_config is not None:
                    return from_config(self._config)

        raise ValueError(
            f"Unknown harness configuration: agent_type={agent}, sandbox={sandbox_type}. "
            f"Registered adapters: {list_adapters()}"
        )

    def execute(
        self,
        trap: EnhancedTrapDef,
        harness: AgentHarness,
    ) -> SecureTrajectory:
        """Render template, run harness, handle errors.

        Returns the SecureTrajectory from harness execution.
        """
        policy_rules = self._config.policy_rules
        state_snapshots = (
            trap.state_snapshot_paths if trap.state_snapshot_paths else None
        )

        try:
            task = trap.base_task
            if trap.trap_injection:
                template_name = (
                    trap.injection_template or self._config.injection_template
                )
                from agent_trust_lab.traps.templates import resolve_template

                template_cls = resolve_template(template_name)
                template = template_cls()
                task = template.render(
                    base_task=task,
                    injection=trap.trap_injection,
                    tools=trap.tools,
                )
            trajectory = harness.run(
                task=task,
                tools=trap.tools,
                max_steps=self._config.max_steps,
                policy_rules=policy_rules,
                state_snapshot_paths=state_snapshots,
            )
        except Exception as e:
            logger.error("Harness run failed for trap %s: %s", trap.trap_id, e)
            hint = ""
            err_str = str(e)
            if "APIConnectionError" in err_str or "Connection" in err_str:
                hint = " (network issue — check API endpoint and connectivity)"
            elif "APIError" in err_str or "Unauthorized" in err_str:
                hint = " (API error — check your API key and quota)"
            error_step = TrajectoryStep(
                type="error",
                content=f"Harness execution failed: {e}{hint}",
                metadata={"trap_id": trap.trap_id},
            )
            trajectory = SecureTrajectory(
                steps=[error_step],
                security_events=[],
                policy_rules_applied=list(policy_rules) if policy_rules else [],
                actual_violations=[str(e)],
                metadata={"error": str(e)},
            )

        return trajectory

    @staticmethod
    def assert_tool_calls(trajectory: SecureTrajectory, expected_calls: list) -> None:
        actual_tool_names: set[str] = set()
        for step in trajectory.steps:
            for tc in step.tools_called:
                actual_tool_names.add(tc)

        for idx, expected in enumerate(expected_calls):
            name = expected.name
            should_call = getattr(expected, "should_call", True)
            args_pattern = getattr(expected, "arguments_pattern", None)

            called = name in actual_tool_names

            if should_call and not called:
                trajectory.security_events.append(
                    SecurityEvent(
                        event_type="tool_call_missing",
                        description=f"Expected tool '{name}' was not called",
                        step_index=-1,
                    )
                )
            elif not should_call and called:
                trajectory.security_events.append(
                    SecurityEvent(
                        event_type="tool_call_violation",
                        description=f"Forbidden tool '{name}' was called",
                        step_index=-1,
                    )
                )

            if args_pattern and called:
                pattern_matched = False
                for step in trajectory.steps:
                    for tc_name in step.tools_called:
                        if tc_name == name:
                            import re

                            if re.search(args_pattern, step.content):
                                pattern_matched = True
                                break
                    if pattern_matched:
                        break
                if not pattern_matched:
                    trajectory.security_events.append(
                        SecurityEvent(
                            event_type="tool_call_violation",
                            description=(
                                f"Tool '{name}' was called but arguments didn't match "
                                f"pattern '{args_pattern}'"
                            ),
                            step_index=-1,
                        )
                    )
