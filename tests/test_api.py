"""Tests for TrustLab and CodeLab Python API wrappers."""

import json
import tempfile
from unittest.mock import patch

from agent_trust_lab.api import CodeLab, TrustLab


class TestTrustLab:
    def test_default_construction(self):
        lab = TrustLab()
        assert lab.config.model == "deepseek-v4-flash"
        assert lab.config.agent_type == "langchain"
        assert lab.config.sandbox == "docker"

    def test_custom_construction(self):
        lab = TrustLab(
            model="custom-model",
            agent_type="langchain",
            base_url="https://custom.api",
            parallel=4,
            skip_hallukg=True,
            calibration_profile="my-profile",
        )
        assert lab.config.model == "custom-model"
        assert lab.config.base_url == "https://custom.api"
        assert lab.config.parallel == 4
        assert lab.config.skip_hallukg is True
        assert lab.config.calibration_profile == "my-profile"

    def test_run_traps(self, sample_traps_dir):
        lab = TrustLab(trap_library_path=sample_traps_dir)
        with (
            patch("agent_trust_lab.llm.get_api_key", return_value=None),
            patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=None),
        ):
            results = lab.run(trap_ids=["test_trap_01"])
            assert len(results) == 1
            assert results[0].trap_id == "test_trap_01"

    def test_run_by_category(self, sample_traps_dir):
        lab = TrustLab(trap_library_path=sample_traps_dir)
        with (
            patch("agent_trust_lab.llm.get_api_key", return_value=None),
            patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=None),
        ):
            results = lab.run(category="general_agent")
            assert len(results) > 0

    def test_list_traps(self, sample_traps_dir):
        lab = TrustLab(trap_library_path=sample_traps_dir)
        traps = lab.list_traps()
        assert len(traps) > 0
        assert "trap_id" in traps[0]

    def test_export_results(self, sample_traps_dir):
        lab = TrustLab(trap_library_path=sample_traps_dir)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            with (
                patch("agent_trust_lab.llm.get_api_key", return_value=None),
                patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=None),
            ):
                results = lab.run(trap_ids=["test_trap_01"])
                lab.export(results, output_path)
            with open(output_path, "r") as f:
                data = json.load(f)
            assert "config" in data
            assert "results" in data
            assert len(data["results"]) == 1
        finally:
            import os

            os.unlink(output_path)

    def test_report_from_json(self):
        lab = TrustLab()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(
                {
                    "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
                    "results": [{"trap_id": "t1", "metadata": {"severity": "medium"}}],
                },
                f,
            )
            json_path = f.name
        try:
            html = lab.report(json_path)
            assert "<!DOCTYPE html>" in html
            assert "t1" in html
        finally:
            import os

            os.unlink(json_path)


class TestCodeLab:
    def test_default_construction(self):
        lab = CodeLab()
        assert lab.config.agent_type == "codex"

    def test_with_codebase(self):
        lab = CodeLab(codebase_path="/path/to/project")
        assert lab.config.codebase_path == "/path/to/project"

    def test_run_code_sets_category(self, sample_traps_dir):
        lab = CodeLab(trap_library_path=sample_traps_dir)
        with (
            patch("agent_trust_lab.llm.get_api_key", return_value=None),
            patch("agent_trust_lab.sandbox.image.get_docker_client", return_value=None),
        ):
            results = lab.run_code(trap_ids=["test_trap_03"])
            assert len(results) == 1
            assert results[0].trap_id == "test_trap_03"
