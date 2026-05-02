"""Integration tests requiring Docker/Podman daemon.

All tests auto-skip when Docker is not available.
"""

import pytest

from agent_trust_lab.models.trajectory import SecureTrajectory, TrajectoryStep

pytestmark = pytest.mark.docker


def _skip_if_no_docker():
    from tests.integration.conftest import _skip_if_missing_docker

    _skip_if_missing_docker()


class TestDockerSandboxIntegration:
    def test_docker_sandbox_safe_command(self):
        _skip_if_no_docker()
        from agent_trust_lab.sandbox.filter import filter_command

        result = filter_command("echo hello")
        assert result.allowed

    def test_code_hallu_checker_real_docker_construction(self):
        _skip_if_no_docker()
        from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

        checker = CodeHalluChecker(timeout=30)
        assert checker.timeout == 30

    def test_code_hallu_real_docker_syntax_error(self):
        _skip_if_no_docker()
        from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

        checker = CodeHalluChecker(timeout=30)
        traj = SecureTrajectory(
            steps=[
                TrajectoryStep(
                    type="code_action",
                    content="```python\nprint('hello'\n```",
                    metadata={},
                )
            ],
            security_events=[],
        )
        reports = checker.batch_check(traj)
        assert len(reports) == 1
        assert reports[0].hallucination_type == "syntax"

    def test_code_hallu_real_docker_attribute_error(self):
        _skip_if_no_docker()
        from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

        checker = CodeHalluChecker(timeout=30)
        traj = SecureTrajectory(
            steps=[
                TrajectoryStep(
                    type="code_action",
                    content="```python\nimport os\nos.nonexistent()\n```",
                    metadata={},
                )
            ],
            security_events=[],
        )
        reports = checker.batch_check(traj)
        assert len(reports) == 1
        assert reports[0].hallucination_type in ("naming", "attribute")

    def test_code_hallu_stub_fallback_when_no_docker(self):
        from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

        checker = CodeHalluChecker(timeout=10)
        traj = SecureTrajectory(
            steps=[
                TrajectoryStep(
                    type="code_action",
                    content="```python\nx = 5\n```",
                    metadata={},
                )
            ],
            security_events=[],
        )
        reports = checker.batch_check(traj)
        assert isinstance(reports, list)
