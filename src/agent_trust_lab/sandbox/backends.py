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
class DockerSandbox(AgentHarness):
    image: str = "agent-trust-lab/sandbox:latest"
    timeout: int = 120
    read_only_mount: Optional[str] = None
    work_dir: str = "/tmp/sandbox"

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="sandbox_init",
                content=(
                    f"Docker sandbox initializing: image={self.image}, "
                    f"timeout={self.timeout}s"
                ),
                metadata={"backend": "docker", "image": self.image},
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
                content=f"Task received: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        steps.append(
            TrajectoryStep(
                type="observation",
                content=(
                    "[DockerSandbox] Stub execution: container would run here. "
                    "No actual Docker execution performed."
                ),
                tools_called=[],
                metadata={"backend": "docker", "status": "stub"},
            )
        )

        if policy_rules:
            policy_violations.extend(policy_rules)

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_violations=policy_violations,
            metadata={"backend": "docker", "image": self.image, "stub": True},
        )


@dataclass
class DryRunSandbox(AgentHarness):
    log_file_path: str = "/tmp/sandbox_dryrun.log"
    intercept_network: bool = True
    intercept_filesystem: bool = True

    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        steps: List[TrajectoryStep] = []
        security_events: List[SecurityEvent] = []
        policy_violations: List[str] = []

        steps.append(
            TrajectoryStep(
                type="sandbox_init",
                content=(
                    "Dry-run sandbox initializing: "
                    "network and filesystem writes will be intercepted."
                ),
                metadata={
                    "backend": "dry-run",
                    "intercept_network": self.intercept_network,
                    "intercept_filesystem": self.intercept_filesystem,
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
                content=f"Task received: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        steps.append(
            TrajectoryStep(
                type="observation",
                content=(
                    "[DryRunSandbox] Stub execution: agent would run here with all "
                    "write/network operations intercepted and logged. "
                    "No actual execution performed."
                ),
                tools_called=[],
                metadata={"backend": "dry-run", "status": "stub"},
            )
        )

        if policy_rules:
            policy_violations.extend(policy_rules)

        dry_run_log = (
            f"[DryRun] Task: {task}\n[DryRun] Tools: {tools}\n"
            f"[DryRun] No writes performed.\n"
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=dry_run_log,
            policy_violations=policy_violations,
            metadata={"backend": "dry-run", "stub": True},
        )
