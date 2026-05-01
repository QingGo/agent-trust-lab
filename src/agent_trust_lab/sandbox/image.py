"""Docker/Podman image management for sandbox execution."""

from __future__ import annotations

from typing import Optional

import docker
from docker.errors import DockerException, ImageNotFound

from agent_trust_lab.log import get_logger

logger = get_logger("sandbox.image")

SANDBOX_LABEL = "agent-trust-lab/sandbox"


def get_docker_client(docker_host: str = "") -> docker.DockerClient:
    """Create a Docker client, auto-detecting Docker or Podman socket.

    Args:
        docker_host: Optional explicit DOCKER_HOST (e.g. unix:///path/to/podman.sock).
    """
    if docker_host:
        return docker.DockerClient(base_url=docker_host)
    try:
        return docker.from_env()
    except DockerException:
        pass
    return docker.DockerClient(base_url="unix:///var/run/docker.sock")


class ImageManager:
    """Manages pull and verification of sandbox container images."""

    def __init__(self, client: Optional[docker.DockerClient] = None):
        self._client = client

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = get_docker_client()
        return self._client

    def ensure_image(self, image_ref: str) -> bool:
        """Pull a container image if not already present locally.

        Returns True if the image is available (already present or successfully pulled).
        """
        if self._image_exists(image_ref):
            return True
        return self._pull_image(image_ref)

    def _image_exists(self, image_ref: str) -> bool:
        try:
            self.client.images.get(image_ref)
            logger.info("Image %s already present", image_ref)
            return True
        except ImageNotFound:
            return False
        except DockerException as e:
            logger.warning("Failed to check image %s: %s", image_ref, e)
            return False

    def _pull_image(self, image_ref: str) -> bool:
        logger.info("Pulling image %s ...", image_ref)
        try:
            self.client.images.pull(image_ref)
            logger.info("Image %s pulled successfully", image_ref)
            return True
        except DockerException as e:
            logger.error("Failed to pull image %s: %s", image_ref, e)
            return False

    def cleanup_orphaned(self) -> int:
        """Remove orphaned sandbox containers from previous runs.

        Returns the number of containers removed.
        """
        try:
            containers = self.client.containers.list(
                all=True,
                filters={"label": SANDBOX_LABEL},
            )
        except DockerException as e:
            logger.warning("Failed to list orphaned containers: %s", e)
            return 0

        count = 0
        for c in containers:
            try:
                c.remove(force=True)
                count += 1
            except DockerException:
                pass

        if count > 0:
            logger.info("Cleaned up %d orphaned sandbox container(s)", count)
        return count
