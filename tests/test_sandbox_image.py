"""Tests for sandbox image management."""

from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException, ImageNotFound

from agent_trust_lab.sandbox.image import SANDBOX_LABEL, ImageManager, get_docker_client


class TestGetDockerClient:
    @patch("agent_trust_lab.sandbox.image.docker.DockerClient")
    def test_explicit_host(self, mock_client_class):
        client = get_docker_client(docker_host="unix:///custom/podman.sock")
        assert client is not None
        mock_client_class.assert_called_once_with(base_url="unix:///custom/podman.sock")

    @patch("agent_trust_lab.sandbox.image.docker.from_env")
    def test_from_env_success(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        client = get_docker_client()
        assert client is mock_client

    @patch(
        "agent_trust_lab.sandbox.image.docker.from_env",
        side_effect=DockerException("no docker"),
    )
    @patch("agent_trust_lab.sandbox.image.docker.DockerClient")
    def test_fallback_to_default_socket(self, mock_client_class, mock_from_env):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        client = get_docker_client()
        assert client is mock_client
        mock_client_class.assert_called_once_with(base_url="unix:///var/run/docker.sock")


class TestImageManager:
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_client):
        return ImageManager(client=mock_client)

    def test_ensure_image_already_present(self, manager, mock_client):
        mock_client.images.get.return_value = MagicMock()
        result = manager.ensure_image("busybox:latest")
        assert result is True
        mock_client.images.get.assert_called_once_with("busybox:latest")
        mock_client.images.pull.assert_not_called()

    def test_ensure_image_pulls_when_missing(self, manager, mock_client):
        mock_client.images.get.side_effect = ImageNotFound("not found")
        mock_client.images.pull.return_value = MagicMock()
        result = manager.ensure_image("busybox:latest")
        assert result is True
        mock_client.images.pull.assert_called_once_with("busybox:latest")

    def test_ensure_image_pull_fails(self, manager, mock_client):
        mock_client.images.get.side_effect = ImageNotFound("not found")
        mock_client.images.pull.side_effect = DockerException("connection refused")
        result = manager.ensure_image("busybox:latest")
        assert result is False

    def test_ensure_image_get_and_pull_both_fail(self, manager, mock_client):
        mock_client.images.get.side_effect = DockerException("daemon down")
        mock_client.images.pull.side_effect = DockerException("pull failed")
        result = manager.ensure_image("busybox:latest")
        assert result is False

    def test_cleanup_orphaned_none_found(self, manager, mock_client):
        mock_client.containers.list.return_value = []
        count = manager.cleanup_orphaned()
        assert count == 0

    def test_cleanup_orphaned_removes_found(self, manager, mock_client):
        c1 = MagicMock()
        c2 = MagicMock()
        mock_client.containers.list.return_value = [c1, c2]
        count = manager.cleanup_orphaned()
        assert count == 2
        c1.remove.assert_called_once_with(force=True)
        c2.remove.assert_called_once_with(force=True)
        mock_client.containers.list.assert_called_once_with(
            all=True, filters={"label": SANDBOX_LABEL}
        )

    def test_cleanup_orphaned_handles_remove_error(self, manager, mock_client):
        c1 = MagicMock()
        c1.remove.side_effect = DockerException("already gone")
        mock_client.containers.list.return_value = [c1]
        count = manager.cleanup_orphaned()
        assert count == 0

    def test_cleanup_orphaned_list_error(self, manager, mock_client):
        mock_client.containers.list.side_effect = DockerException("daemon down")
        count = manager.cleanup_orphaned()
        assert count == 0

    def test_client_property_lazy_init(self):
        with patch("agent_trust_lab.sandbox.image.get_docker_client") as mock_get:
            mock_client = MagicMock()
            mock_get.return_value = mock_client
            manager = ImageManager()
            manager.cleanup_orphaned()
            mock_get.assert_called_once()
