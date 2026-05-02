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

logger = get_logger("sandbox.backends")


@register_adapter("docker")
@dataclass
class DockerSandbox(AgentHarness):
    image: str = "docker.m.daocloud.io/library/busybox:latest"
    timeout: int = 120
    read_only_mount: Optional[str] = None
    work_dir: str = "/tmp/sandbox"
    network_enabled: bool = False
    tmpfs_size: str = "64m"
    docker_host: str = ""

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "DockerSandbox":
        return cls(
            image=config.sandbox_image or "docker.m.daocloud.io/library/busybox:latest",
            timeout=config.timeout,
            network_enabled=config.sandbox_network,
            tmpfs_size=config.sandbox_tmpfs_size,
            docker_host=config.docker_host,
        )

    def run(
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
                type="sandbox_init",
                content=(
                    f"Docker sandbox initializing: image={self.image}, timeout={self.timeout}s"
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
            return SecureTrajectory(
                steps=steps,
                security_events=security_events,
                dry_run_log="",
                policy_rules_applied=policy_rules_applied,
                actual_violations=actual_violations,
                metadata={"backend": "docker", "image": self.image, "stub": False},
            )

        try:
            return self._execute_in_container(
                task=task,
                tools=tools,
                steps=steps,
                security_events=security_events,
                policy_rules_applied=policy_rules_applied,
                actual_violations=actual_violations,
            )
        except Exception as e:
            logger.warning("Container execution failed: %s", e)
            return self._fallback_stub(
                task=task,
                steps=steps,
                security_events=security_events,
                policy_rules_applied=policy_rules_applied,
                actual_violations=actual_violations,
                error=str(e),
            )

    def _execute_in_container(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        steps: List[TrajectoryStep],
        security_events: List[SecurityEvent],
        policy_rules_applied: List[str],
        actual_violations: List[str],
    ) -> SecureTrajectory:
        from agent_trust_lab.sandbox.image import SANDBOX_LABEL, ImageManager, get_docker_client

        client = get_docker_client(self.docker_host)
        img_mgr = ImageManager(client)
        img_mgr.cleanup_orphaned()

        if not img_mgr.ensure_image(self.image):
            return self._fallback_stub(
                task=task,
                steps=steps,
                security_events=security_events,
                policy_rules_applied=policy_rules_applied,
                actual_violations=actual_violations,
                error=f"Failed to pull image: {self.image}",
            )

        steps.append(
            TrajectoryStep(
                type="thought",
                content=f"Executing task in sandbox: {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        security_opts = {
            "read_only": True,
            "cap_drop": ["ALL"],
            "tmpfs": {"/tmp": f"size={self.tmpfs_size}"},
            "mem_limit": "128m",
            "nano_cpus": 500_000_000,
            "working_dir": self.work_dir,
        }

        if not self.network_enabled:
            security_opts["network_disabled"] = True

        container = None
        try:
            container = client.containers.run(
                image=self.image,
                command=["sh", "-c", task],
                detach=True,
                auto_remove=True,
                labels={SANDBOX_LABEL: ""},
                **security_opts,
            )

            result = container.wait(timeout=self.timeout)
            exit_code = result.get("StatusCode", -1)

            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

            if exit_code != 0:
                actual_violations.append(
                    f"Container exited with code {exit_code}"
                )

            steps.append(
                TrajectoryStep(
                    type="observation",
                    content=logs[:2000] if logs else "(no output)",
                    metadata={
                        "backend": "docker",
                        "exit_code": exit_code,
                        "status": "executed",
                    },
                )
            )

        except Exception as e:
            error_msg = str(e)[:500]
            if container:
                try:
                    container.kill()
                except Exception:
                    pass
            actual_violations.append(error_msg)
            steps.append(
                TrajectoryStep(
                    type="observation",
                    content=f"[DockerSandbox] Execution error: {error_msg}",
                    metadata={"backend": "docker", "status": "error"},
                )
            )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={
                "backend": "docker",
                "image": self.image,
                "stub": False,
            },
        )

    def _fallback_stub(
        self,
        task: str,
        steps: List[TrajectoryStep],
        security_events: List[SecurityEvent],
        policy_rules_applied: List[str],
        actual_violations: List[str],
        error: str = "",
    ) -> SecureTrajectory:
        fallback_msg = (
            "[DockerSandbox] Stub execution: container would run here. "
            "No actual Docker execution performed."
        )
        if error:
            fallback_msg += f" (Error: {error[:200]})"

        steps.append(
            TrajectoryStep(
                type="thought",
                content=f"Task received (stub fallback): {task[:200]}",
                metadata={"task_length": len(task)},
            )
        )

        steps.append(
            TrajectoryStep(
                type="observation",
                content=fallback_msg,
                tools_called=[],
                metadata={"backend": "docker", "status": "stub"},
            )
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log="",
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={"backend": "docker", "image": self.image, "stub": True},
        )


@register_adapter("dry-run")
@dataclass
class DryRunSandbox(AgentHarness):
    log_file_path: str = "/tmp/sandbox_dryrun.log"
    intercept_network: bool = True
    intercept_filesystem: bool = True

    @classmethod
    def from_config(cls, config: "EvaluationConfig") -> "DryRunSandbox":
        return cls()

    def run(
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

        dry_run_log = (
            f"[DryRun] Task: {task}\n[DryRun] Tools: {tools}\n[DryRun] No writes performed.\n"
        )

        return SecureTrajectory(
            steps=steps,
            security_events=security_events,
            dry_run_log=dry_run_log,
            policy_rules_applied=policy_rules_applied,
            actual_violations=actual_violations,
            metadata={"backend": "dry-run", "stub": True},
        )
