"""Tests for batch evaluation module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_trust_lab.batch import (
    BatchConfig,
    EvaluationSpec,
    parse_batch_yaml,
    run_batch,
)


class TestParseBatchYaml:
    def test_parse_minimal_config(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "test-eval"
    traps:
      category: "general_agent"
""")
        cfg = parse_batch_yaml(str(yaml_path))
        assert len(cfg.evaluations) == 1
        assert cfg.evaluations[0].label == "test-eval"
        assert cfg.evaluations[0].model == "deepseek-v4-flash"
        assert cfg.evaluations[0].agent_type == "langchain"
        assert cfg.evaluations[0].traps == {"category": "general_agent"}

    def test_parse_full_config(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "flash-baseline"
    model: "deepseek-v4-flash"
    agent_type: "langchain"
    thinking_enabled: false
    traps:
      category: "general_agent"
      limit: 5
  - label: "flash-thinking"
    model: "deepseek-v4-flash"
    agent_type: "langchain"
    thinking_enabled: true
    reasoning_effort: "high"
    traps:
      trap_ids: ["test_01", "test_02"]
      mutate: true
common:
  sandbox: "docker"
  sandbox_image: "alpine:latest"
  parallel: 2
  output_dir: "./batch-out/"
report:
  format: "markdown"
  lang: "zh"
  open: true
""")
        cfg = parse_batch_yaml(str(yaml_path))

        assert len(cfg.evaluations) == 2
        assert cfg.evaluations[0].label == "flash-baseline"
        assert cfg.evaluations[0].thinking_enabled is False
        assert cfg.evaluations[1].label == "flash-thinking"
        assert cfg.evaluations[1].thinking_enabled is True
        assert cfg.evaluations[1].reasoning_effort == "high"
        assert cfg.evaluations[1].traps == {
            "trap_ids": ["test_01", "test_02"],
            "mutate": True,
        }
        assert cfg.sandbox == "docker"
        assert cfg.sandbox_image == "alpine:latest"
        assert cfg.parallel == 2
        assert cfg.output_dir == "./batch-out/"
        assert cfg.report_format == "markdown"
        assert cfg.report_lang == "zh"
        assert cfg.report_open is True

    def test_parse_defaults(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "default-test"
    traps:
      limit: 3
""")
        cfg = parse_batch_yaml(str(yaml_path))
        assert cfg.trap_library_path == ""
        assert cfg.sandbox == "docker"
        assert cfg.sandbox_image == ""
        assert cfg.sandbox_network is False
        assert cfg.docker_host == ""
        assert cfg.parallel == 1
        assert cfg.output_dir == "./results/"
        assert cfg.report_format == "html"
        assert cfg.report_lang == "en"
        assert cfg.report_open is False
        assert cfg.calibration_profile is None
        assert cfg.timeout == 120

    def test_parse_explicit_trap_ids(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "by-ids"
    traps:
      trap_ids: ["hallu_check_01", "info_leak_01"]
""")
        cfg = parse_batch_yaml(str(yaml_path))
        assert cfg.evaluations[0].traps == {
            "trap_ids": ["hallu_check_01", "info_leak_01"],
        }

    def test_parse_missing_evaluations(self, tmp_path):
        yaml_path = tmp_path / "no_evals.yaml"
        yaml_path.write_text("common:\n  parallel: 2\n")
        with pytest.raises(ValueError, match="evaluations"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_empty_evaluations(self, tmp_path):
        yaml_path = tmp_path / "empty_evals.yaml"
        yaml_path.write_text("evaluations: []\n")
        with pytest.raises(ValueError, match="non-empty"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_evaluations_not_a_list(self, tmp_path):
        yaml_path = tmp_path / "bad_evals.yaml"
        yaml_path.write_text("evaluations: 42\n")
        with pytest.raises(ValueError, match="non-empty"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_missing_label(self, tmp_path):
        yaml_path = tmp_path / "no_label.yaml"
        yaml_path.write_text("""
evaluations:
  - model: "test-model"
    traps:
      category: "test"
""")
        with pytest.raises(ValueError, match="label"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_eval_not_a_mapping(self, tmp_path):
        yaml_path = tmp_path / "bad_eval.yaml"
        yaml_path.write_text("""
evaluations:
  - "not-a-mapping"
""")
        with pytest.raises(ValueError, match="mapping"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_traps_not_a_mapping(self, tmp_path):
        yaml_path = tmp_path / "bad_traps.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "test"
    traps: "not-a-mapping"
""")
        with pytest.raises(ValueError, match="traps"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_not_a_dict(self, tmp_path):
        yaml_path = tmp_path / "list.yaml"
        yaml_path.write_text("- evaluations: []\n")
        with pytest.raises(ValueError, match="mapping"):
            parse_batch_yaml(str(yaml_path))

    def test_parse_boolean_values(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "bool-test"
    thinking_enabled: true
    skip_hallukg: false
    traps:
      category: "test"
common:
  sandbox_network: true
report:
  open: false
""")
        cfg = parse_batch_yaml(str(yaml_path))
        assert cfg.evaluations[0].thinking_enabled is True
        assert cfg.evaluations[0].skip_hallukg is False
        assert cfg.sandbox_network is True
        assert cfg.report_open is False

    def test_parse_calibration_profile(self, tmp_path):
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "calib-test"
    traps:
      trap_ids: ["test_01"]
common:
  calibration_profile: "my-profile"
""")
        cfg = parse_batch_yaml(str(yaml_path))
        assert cfg.calibration_profile == "my-profile"


class TestBatchConfig:
    def test_evaluation_spec_defaults(self):
        spec = EvaluationSpec(label="test")
        assert spec.label == "test"
        assert spec.model == "deepseek-v4-flash"
        assert spec.agent_type == "langchain"
        assert spec.thinking_enabled is False
        assert spec.reasoning_effort == ""
        assert spec.traps == {}

    def test_batch_config_defaults(self):
        cfg = BatchConfig(evaluations=[EvaluationSpec(label="t1")])
        assert len(cfg.evaluations) == 1
        assert cfg.sandbox == "docker"
        assert cfg.parallel == 1
        assert cfg.report_format == "html"
        assert cfg.report_lang == "en"


class TestRunBatch:
    @pytest.fixture(autouse=True)
    def _stub_hallukg(self):
        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            yield

    @pytest.fixture
    def batch_trap_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            general_dir = Path(tmpdir) / "general"
            general_dir.mkdir()
            (general_dir / "batch_trap_01.yaml").write_text(
                """trap_id: batch_trap_01
trap_type: parameter_hallucination
version: "1.0.0"
severity: medium
difficulty: medium
category: general_agent
base_task: "Test task for batch."
tools:
  - name: shell
trap_injection: "Use fake parameter."
variation_rules: []
""",
                encoding="utf-8",
            )
            (general_dir / "batch_trap_02.yaml").write_text(
                """trap_id: batch_trap_02
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Read a file."
tools:
  - name: file_read
trap_injection: ""
variation_rules: []
""",
                encoding="utf-8",
            )
            yield tmpdir

    def test_run_batch_two_configs(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="model-a",
                    model="deepseek-test-a",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
                EvaluationSpec(
                    label="model-b",
                    model="deepseek-test-b",
                    thinking_enabled=True,
                    reasoning_effort="high",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)

        assert "configs" in merged
        assert len(merged["configs"]) == 2
        assert merged["configs"][0]["model"] == "deepseek-test-a"
        assert merged["configs"][1]["model"] == "deepseek-test-b"
        assert merged["configs"][1]["thinking_enabled"] is True

        assert "results" in merged
        assert len(merged["results"]) == 1
        assert merged["results"][0]["trap_id"] == "batch_trap_01"
        assert "scores" in merged["results"][0]
        assert len(merged["results"][0]["scores"]) == 2

        assert os.path.isfile(os.path.join(output_dir, "model-a.json"))
        assert os.path.isfile(os.path.join(output_dir, "model-b.json"))
        assert os.path.isfile(os.path.join(output_dir, "comparison.json"))
        assert os.path.isfile(os.path.join(output_dir, "comparison.html"))

    def test_run_batch_single_config(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_single")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="solo",
                    model="deepseek-solo",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)

        assert "configs" in merged
        assert len(merged["configs"]) == 1
        assert os.path.isfile(os.path.join(output_dir, "solo.json"))
        assert os.path.isfile(os.path.join(output_dir, "comparison.html"))

    def test_run_batch_markdown_report(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_md")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="md-test-a",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
                EvaluationSpec(
                    label="md-test-b",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
            report_format="markdown",
        )

        run_batch(cfg)
        assert os.path.isfile(os.path.join(output_dir, "comparison.md"))
        content = Path(os.path.join(output_dir, "comparison.md")).read_text()
        assert "# Agent Trust Evaluation Report" in content

    def test_run_batch_zh_report(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_zh")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="zh-a",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
                EvaluationSpec(
                    label="zh-b",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
            report_lang="zh",
        )

        run_batch(cfg)
        content = Path(os.path.join(output_dir, "comparison.html")).read_text()
        assert "可信" in content

    def test_run_batch_by_category(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_cat")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="cat-test-a",
                    traps={"category": "general_agent", "limit": 1},
                ),
                EvaluationSpec(
                    label="cat-test-b",
                    traps={"category": "general_agent", "limit": 1},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)
        assert len(merged["results"]) == 1

    def test_run_batch_multiple_traps(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_multi")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="multi-a",
                    traps={"category": "general_agent"},
                ),
                EvaluationSpec(
                    label="multi-b",
                    traps={"category": "general_agent"},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)
        n = len(merged["results"])
        assert n == 2
        trap_ids = {r["trap_id"] for r in merged["results"]}
        assert "batch_trap_01" in trap_ids
        assert "batch_trap_02" in trap_ids

    def test_run_batch_json_structure(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_struct")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="struct-a",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
                EvaluationSpec(
                    label="struct-b",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)

        for result in merged["results"]:
            assert "trap_id" in result
            assert "scores" in result
            for label, entry in result["scores"].items():
                assert "compliance" in entry
                assert "hallucination" in entry
                assert "metadata" in entry

        for cfg_entry in merged["configs"]:
            assert "model" in cfg_entry
            assert "thinking_enabled" in cfg_entry
            assert "config_label" in cfg_entry

    def test_run_batch_safe_label(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_safe")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="my model/v2 test",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
                EvaluationSpec(
                    label="other-config",
                    traps={"trap_ids": ["batch_trap_01"]},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        run_batch(cfg)
        assert os.path.isfile(os.path.join(output_dir, "my_model_v2_test.json"))

    def test_run_batch_empty_traps_runs_all(self, batch_trap_dir):
        output_dir = os.path.join(batch_trap_dir, "output_all_empty")
        cfg = BatchConfig(
            evaluations=[
                EvaluationSpec(
                    label="all-a",
                    traps={},
                ),
                EvaluationSpec(
                    label="all-b",
                    traps={},
                ),
            ],
            trap_library_path=batch_trap_dir,
            output_dir=output_dir,
        )

        merged = run_batch(cfg)
        assert len(merged["results"]) == 2


class TestBatchCLI:
    def test_batch_command_help(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.stdout.lower()

    def test_batch_command_missing_config(self):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["batch", "/nonexistent/batch.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_batch_command_invalid_yaml(self, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("evaluations: []\n")
        result = runner.invoke(app, ["batch", str(yaml_path)])
        assert result.exit_code == 1
        assert "non-empty" in result.stdout

    def test_batch_command_invalid_format(self, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        yaml_path = tmp_path / "bad_fmt.yaml"
        yaml_path.write_text("""
evaluations:
  - label: "test"
    traps:
      trap_ids: ["test_01"]
report:
  format: "xml"
""")
        result = runner.invoke(app, ["batch", str(yaml_path)])
        assert result.exit_code == 1
        assert "Invalid report format" in result.stdout

    @patch("agent_trust_lab.llm.get_api_key", return_value=None)
    def test_batch_command_success(self, _mock_key, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: parameter_hallucination
version: "1.0.0"
severity: medium
difficulty: medium
category: general_agent
base_task: "Test."
tools: [{name: shell}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text(
            f"""evaluations:
  - label: "cli-a"
    model: "test-model-a"
    traps:
      trap_ids: ["t1"]
  - label: "cli-b"
    model: "test-model-b"
    traps:
      trap_ids: ["t1"]
common:
  trap_library_path: "{trap_dir}"
  output_dir: "{output_dir}"
report:
  format: "html"
"""
        )

        result = runner.invoke(app, ["batch", str(yaml_path)])
        assert result.exit_code == 0
        assert "Batch complete" in result.stdout

        assert os.path.isfile(os.path.join(str(output_dir), "cli-a.json"))
        assert os.path.isfile(os.path.join(str(output_dir), "cli-b.json"))
        assert os.path.isfile(os.path.join(str(output_dir), "comparison.json"))
        assert os.path.isfile(os.path.join(str(output_dir), "comparison.html"))

    @patch("agent_trust_lab.llm.get_api_key", return_value=None)
    def test_batch_command_markdown_output(self, _mock_key, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Read file."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text(
            f"""evaluations:
  - label: "md-test-a"
    traps:
      trap_ids: ["t1"]
  - label: "md-test-b"
    traps:
      trap_ids: ["t1"]
common:
  trap_library_path: "{trap_dir}"
  output_dir: "{output_dir}"
report:
  format: "markdown"
  lang: "en"
"""
        )

        result = runner.invoke(app, ["batch", str(yaml_path)])
        assert result.exit_code == 0
        assert os.path.isfile(os.path.join(str(output_dir), "comparison.md"))

    @patch("agent_trust_lab.llm.get_api_key", return_value=None)
    def test_batch_command_zh_report(self, _mock_key, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()

        trap_dir = tmp_path / "traps" / "general"
        trap_dir.mkdir(parents=True)
        (trap_dir / "t1.yaml").write_text(
            """trap_id: t1
trap_type: benign_control
version: "1.0.0"
severity: none
difficulty: trivial
category: general_agent
base_task: "Read file."
tools: [{name: file_read}]
trap_injection: ""
variation_rules: []
""",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        yaml_path = tmp_path / "batch.yaml"
        yaml_path.write_text(
            f"""evaluations:
  - label: "zh-test-a"
    traps:
      trap_ids: ["t1"]
  - label: "zh-test-b"
    traps:
      trap_ids: ["t1"]
common:
  trap_library_path: "{trap_dir}"
  output_dir: "{output_dir}"
report:
  format: "html"
  lang: "zh"
"""
        )

        result = runner.invoke(app, ["batch", str(yaml_path)])
        assert result.exit_code == 0
        with open(os.path.join(str(output_dir), "comparison.html")) as f:
            assert "可信" in f.read()
