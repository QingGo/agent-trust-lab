from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_trust_lab.adapters.registry import register_adapter
from agent_trust_lab.core.protocols import ContainerRuntime
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
    container_runtime: Optional[ContainerRuntime] = field(default=None, repr=False)

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
        state_snapshot_paths: Optional[List[str]] = None,
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
                state_snapshot_paths=state_snapshot_paths,
            )
        except Exception as e:
            logger.warning(
                "Container execution failed (falling back to stub): %s. "
                "Check Docker daemon status and container limits.",
                e,
            )
            return self._fallback_stub(
                task=task,
                steps=steps,
                security_events=security_events,
                policy_rules_applied=policy_rules_applied,
                actual_violations=actual_violations,
                error=str(e),
            )

    def _get_runtime(self) -> ContainerRuntime:
        """Return the configured container runtime, creating a default if needed."""
        if self.container_runtime is not None:
            return self.container_runtime
        from agent_trust_lab.sandbox.runtime import DockerContainerRuntime

        return DockerContainerRuntime(docker_host=self.docker_host)

    def _execute_in_container(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        steps: List[TrajectoryStep],
        security_events: List[SecurityEvent],
        policy_rules_applied: List[str],
        actual_violations: List[str],
        state_snapshot_paths: Optional[List[str]] = None,
    ) -> SecureTrajectory:
        from agent_trust_lab.sandbox.image import SANDBOX_LABEL

        runtime = self._get_runtime()
        runtime.cleanup_orphaned(SANDBOX_LABEL)

        if not runtime.ensure_image(self.image):
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

        pre_snapshot: dict[str, str] = {}
        if state_snapshot_paths:
            pre_snapshot = self._take_state_snapshot(runtime, state_snapshot_paths)

        try:
            exit_code, stdout, _stderr = runtime.run(
                image=self.image,
                command=["sh", "-c", task],
                timeout=self.timeout,
                network_enabled=self.network_enabled,
                tmpfs_size=self.tmpfs_size,
                work_dir=self.work_dir,
                labels={SANDBOX_LABEL: ""},
            )

            logs = stdout
            if exit_code != 0:
                actual_violations.append(f"Container exited with code {exit_code}")

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

            if pre_snapshot:
                for path, pre_hash in pre_snapshot.items():
                    post_hash = self._get_file_hash(runtime, path)
                    if post_hash is not None and post_hash != pre_hash:
                        security_events.append(
                            SecurityEvent(
                                event_type="state_diff_detected",
                                description=(
                                    f"File {path} hash changed: "
                                    f"{pre_hash[:16]}... -> {post_hash[:16]}..."
                                ),
                                step_index=len(steps) - 1,
                            )
                        )

        except Exception as e:
            error_msg = str(e)[:500]
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

    @staticmethod
    def _get_file_hash(runtime: ContainerRuntime, path: str) -> Optional[str]:
        if not runtime.ensure_image("docker.m.daocloud.io/library/busybox:latest"):
            return None
        try:
            exit_code, stdout, _stderr = runtime.run(
                image="docker.m.daocloud.io/library/busybox:latest",
                command=["sha256sum", path],
                timeout=10,
                network_enabled=False,
                mem_limit="32m",
                labels={
                    "agent-trust-lab/sandbox": "",
                },
            )
            if exit_code != 0:
                return None
            parts = stdout.strip().split()
            if parts:
                return parts[0]
        except Exception as e:
            logger.warning("State snapshot hash failed for %s: %s", path, e)
        return None

    @staticmethod
    def _take_state_snapshot(
        runtime: ContainerRuntime, paths: List[str]
    ) -> Dict[str, str]:
        snap: Dict[str, str] = {}
        _hash = DockerSandbox._get_file_hash  # pyright: ignore[reportAttributeAccessIssue]
        for path in paths:
            h = _hash(runtime, path)
            if h:
                snap[path] = h
        return snap

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
        state_snapshot_paths: Optional[List[str]] = None,
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
