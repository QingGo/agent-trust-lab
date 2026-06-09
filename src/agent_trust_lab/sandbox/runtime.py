"""Container runtime implementations (Docker and stub) for sandbox execution.

These satisfy the ContainerRuntime protocol (agent_trust_lab.core.protocols),
enabling DI into DockerSandbox for testability and future Podman support.
"""

from __future__ import annotations

from typing import Any

from agent_trust_lab.log import get_logger

logger = get_logger("sandbox.runtime")


class DockerContainerRuntime:
    """Container runtime backed by a local Docker daemon.

    Wraps docker.from_env() + ImageManager for full lifecycle:
    image pull/verify/cleanup + container run + log collection.

    Satisfies ContainerRuntime protocol.
    """

    def __init__(self, docker_host: str = "") -> None:
        self._docker_host = docker_host

    def run(
        self,
        image: str,
        command: list[str],
        *,
        timeout: int = 30,
        network_enabled: bool = False,
        tmpfs_size: str = "64m",
        work_dir: str = "/tmp/sandbox",
        labels: dict[str, str] | None = None,
        read_only: bool = True,
        mem_limit: str = "128m",
        **kwargs: Any,
    ) -> tuple[int, str, str]:
        from agent_trust_lab.sandbox.image import SANDBOX_LABEL, get_docker_client

        client = get_docker_client(self._docker_host)

        security_opts: dict[str, Any] = {
            "read_only": read_only,
            "privileged": False,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "tmpfs": {"/tmp": f"size={tmpfs_size}"},
            "mem_limit": mem_limit,
            "working_dir": work_dir,
        }

        if not network_enabled:
            security_opts["network_disabled"] = True

        security_opts.update(kwargs)

        effective_labels = {SANDBOX_LABEL: ""}
        if labels:
            effective_labels.update(labels)

        container = client.containers.run(
            image=image,
            command=command,
            detach=True,
            auto_remove=True,
            labels=effective_labels,
            **security_opts,
        )

        try:
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", -1)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            return exit_code, logs, ""
        except Exception:
            try:
                container.kill()
            except Exception as e:
                logger.warning("Failed to kill sandbox container: %s", e)
            raise

    def ensure_image(self, image_ref: str) -> bool:
        from agent_trust_lab.sandbox.image import ImageManager, get_docker_client

        client = get_docker_client(self._docker_host)
        return ImageManager(client).ensure_image(image_ref)

    def cleanup_orphaned(self, label: str) -> int:
        from agent_trust_lab.sandbox.image import ImageManager, get_docker_client

        client = get_docker_client(self._docker_host)
        return ImageManager(client).cleanup_orphaned()


class StubContainerRuntime:
    """No-op container runtime for testing without Docker.

    Returns success for all operations. Satisfies ContainerRuntime protocol.
    """

    def run(
        self,
        image: str,
        command: list[str],
        *,
        timeout: int = 30,
        network_enabled: bool = False,
        tmpfs_size: str = "64m",
        work_dir: str = "/tmp/sandbox",
        labels: dict[str, str] | None = None,
        read_only: bool = True,
        mem_limit: str = "128m",
        **kwargs: Any,
    ) -> tuple[int, str, str]:
        return 0, "[StubContainerRuntime] command would run here", ""

    def ensure_image(self, image_ref: str) -> bool:
        return True

    def cleanup_orphaned(self, label: str) -> int:
        return 0
