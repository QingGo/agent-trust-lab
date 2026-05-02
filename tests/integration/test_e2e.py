"""End-to-end tests: single trap → real LLM → full orchestrator pipeline.

All tests auto-skip when DEEPSEEK_API_KEY is not set.
"""

import pytest

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.orchestrator import Orchestrator

pytestmark = pytest.mark.integration


def _skip_if_no_key():
    from tests.integration.conftest import _skip_if_missing_api

    _skip_if_missing_api()


class TestSingleTrapE2E:
    def test_benign_control_langchain(self, benign_trap_dir):
        _skip_if_no_key()
        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=benign_trap_dir,
            max_steps=3,
            skip_hallukg=False,
        )
        orch = Orchestrator(config)
        results = orch.run_traps(category="general_agent", limit=1)
        assert len(results) == 1
        result = results[0]
        assert result.trap_id == "test_benign_control_01"
        assert result.trap_type == "benign_control"
        assert result.category == "general_agent"
        assert len(result.trajectory.steps) >= 2
        assert result.compliance is not None
        assert result.error is None

    def test_orchestrator_run_single_stub_fallback(self, benign_trap_dir):
        _skip_if_no_key()
        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=benign_trap_dir,
            max_steps=3,
            skip_hallukg=True,
        )
        orch = Orchestrator(config)
        results = orch.run_traps(category="general_agent", limit=1)
        assert len(results) == 1
        result = results[0]
        assert result.trajectory is not None
        assert result.compliance is not None
        dims = result.compliance.dimensions
        assert "tool_authorization" in dims or "state_consistency" in dims

    def test_export_results_has_valid_json(self, benign_trap_dir, tmp_path):
        _skip_if_no_key()
        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=benign_trap_dir,
            max_steps=2,
            skip_hallukg=True,
        )
        orch = Orchestrator(config)
        results = orch.run_traps(category="general_agent", limit=1)
        output = tmp_path / "e2e_output.json"
        orch.export_results(results, str(output))
        assert output.exists()
        import json

        data = json.loads(output.read_text(encoding="utf-8"))
        assert "config" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["trap_id"] == "test_benign_control_01"

    def test_skip_hallukg_flag(self, benign_trap_dir):
        _skip_if_no_key()
        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=benign_trap_dir,
            max_steps=2,
            skip_hallukg=True,
        )
        orch = Orchestrator(config)
        results = orch.run_traps(category="general_agent", limit=1)
        assert len(results) == 1
        assert results[0].hallucination_steps == []

    def test_replay_trajectory(self, benign_trap_dir):
        _skip_if_no_key()
        config = EvaluationConfig(
            agent_type="langchain",
            model="deepseek-v4-flash",
            sandbox="dry-run",
            trap_library_path=benign_trap_dir,
            max_steps=2,
            skip_hallukg=True,
        )
        orch = Orchestrator(config)
        results = orch.run_traps(category="general_agent", limit=1)
        original = results[0]

        replayed = orch.replay_trajectory(
            trajectory=original.trajectory,
            trap_id=original.trap_id,
            trap_type=original.trap_type,
            category=original.category,
            knowledge_source="The file_read tool reads files.",
        )
        assert replayed.trap_id == original.trap_id
        assert replayed.compliance is not None
