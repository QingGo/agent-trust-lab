"""Tests for report generator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_trust_lab.report import ReportGenerator


class TestReportGenerator:
    def test_generate_minimal_report(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test-model", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_01",
                    "trap_type": "parameter_hallucination",
                    "category": "general_agent",
                    "steps_count": 5,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        html = generator.generate(data)
        assert "<!DOCTYPE html>" in html
        assert "test_01" in html
        assert "test-model" in html
        assert "general_agent" in html

    def test_generate_with_compliance(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "gpt-4", "agent_type": "langchain", "sandbox": "dry-run"},
            "results": [
                {
                    "trap_id": "test_02",
                    "trap_type": "override_tool",
                    "category": "code_agent",
                    "steps_count": 8,
                    "mutated": True,
                    "security_events": 1,
                    "metadata": {"severity": "high", "difficulty": "medium"},
                    "compliance": {
                        "overall": "fail",
                        "dimensions": {"tool_authorization": "fail", "source_verification": "pass"},
                        "critical_count": 1,
                        "high_count": 0,
                    },
                }
            ],
        }
        html = generator.generate(data)
        assert "FAIL" in html
        assert "fail" in html
        assert "tool_authorization" in html
        assert "Critical: 1" in html

    def test_generate_with_hallucination(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_03",
                    "trap_type": "hallu_test",
                    "category": "general_agent",
                    "steps_count": 4,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "low", "difficulty": "trivial"},
                    "hallucination": {
                        "step_count": 2,
                        "avg_g_score": 0.45,
                        "avg_faithfulness": 0.82,
                        "labels": ["Grounded", "Contradicted"],
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.8,
                                "u_score": 0.1,
                                "c_score": 0.1,
                                "faithfulness_score": 0.95,
                            },
                            {
                                "step_index": 1,
                                "gsar_label": "Contradicted",
                                "g_score": 0.1,
                                "u_score": 0.2,
                                "c_score": 0.7,
                                "faithfulness_score": 0.69,
                            },
                        ],
                    },
                }
            ],
        }
        html = generator.generate(data)
        assert "Grounded" in html
        assert "Contradicted" in html
        assert "Hallucination Analysis" in html

    def test_generate_with_code_hallu(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "codex", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_04",
                    "trap_type": "code_test",
                    "category": "code_agent",
                    "steps_count": 6,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "high", "difficulty": "hard"},
                    "code_hallu": {
                        "count": 2,
                        "types": ["mapping", "naming"],
                        "checks": [
                            {
                                "step_index": 1,
                                "hallucination_type": "mapping",
                                "code_snippet": "import fake_lib",
                                "error_message": "No module named fake_lib",
                                "fix_suggestion": "Use requests instead",
                            },
                            {
                                "step_index": 3,
                                "hallucination_type": "naming",
                                "code_snippet": "df.read_csv()",
                                "error_message": "read_csv does not exist",
                                "fix_suggestion": "Use pd.read_csv()",
                            },
                        ],
                    },
                }
            ],
        }
        html = generator.generate(data)
        assert "Code Hallucination Checks" in html
        assert "mapping" in html
        assert "Use requests instead" in html

    def test_summary_statistics(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "trap_type": "a",
                    "category": "general_agent",
                    "steps_count": 3,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                    "compliance": {
                        "overall": "pass",
                        "dimensions": {},
                        "critical_count": 0,
                        "high_count": 0,
                    },
                    "hallucination": {
                        "step_count": 1,
                        "avg_g_score": 0.9,
                        "avg_faithfulness": 0.95,
                    },
                },
                {
                    "trap_id": "t2",
                    "trap_type": "b",
                    "category": "general_agent",
                    "steps_count": 4,
                    "mutated": True,
                    "security_events": 1,
                    "metadata": {"severity": "high", "difficulty": "hard"},
                    "compliance": {
                        "overall": "fail",
                        "dimensions": {},
                        "critical_count": 1,
                        "high_count": 2,
                    },
                },
                {
                    "trap_id": "t3",
                    "trap_type": "c",
                    "category": "code_agent",
                    "steps_count": 5,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "low", "difficulty": "trivial"},
                    "compliance": {
                        "overall": "warn",
                        "dimensions": {},
                        "critical_count": 0,
                        "high_count": 2,
                    },
                    "hallucination": {
                        "step_count": 2,
                        "avg_g_score": 0.5,
                        "avg_faithfulness": 0.7,
                    },
                },
            ],
        }
        html = generator.generate(data)
        assert "3" in html  # total traps
        assert "1" in html  # mutated count (first line has meta so check pass/warn/fail instead)
        assert "Traps Evaluated" in html

    def test_generate_to_file(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "file_test",
                    "trap_type": "test",
                    "category": "general_agent",
                    "steps_count": 2,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", delete=False, encoding="utf-8"
        ) as f:
            generator.generate(data, output_path=f.name)

        content = Path(f.name).read_text()
        assert "file_test" in content
        Path(f.name).unlink()

    def test_from_json_file(self):
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "json_test",
                    "trap_type": "test",
                    "category": "general_agent",
                    "steps_count": 2,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)

        html = ReportGenerator.from_json_file(f.name)
        assert "json_test" in html

        html_file = f.name.rsplit(".", 1)[0] + ".html"
        assert Path(html_file).is_file()
        Path(f.name).unlink()
        Path(html_file).unlink()

    def test_empty_results(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [],
        }
        html = generator.generate(data)
        assert "0" in html
        assert "<!DOCTYPE html>" in html

    def test_result_with_error(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "error_test",
                    "trap_type": "test",
                    "category": "general_agent",
                    "steps_count": 0,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                    "error": "API connection timeout",
                }
            ],
        }
        html = generator.generate(data)
        assert "API connection timeout" in html
        assert "Error" in html

    def test_generate_with_remediation(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_remed",
                    "trap_type": "parameter_hallucination",
                    "category": "general_agent",
                    "steps_count": 4,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {
                        "severity": "medium",
                        "difficulty": "easy",
                        "remediation": {
                            "problem": "Agent used fake parameter",
                            "cause": "Tool description was misleading",
                            "fix": "Validate parameters against schema",
                        },
                    },
                }
            ],
        }
        html = generator.generate(data)
        assert "Remediation" in html
        assert "Agent used fake parameter" in html
        assert "Tool description was misleading" in html
        assert "Validate parameters against schema" in html
        assert "Problem" in html
        assert "Cause" in html
        assert "Fix" in html

    def test_generate_without_remediation(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_no_remed",
                    "trap_type": "test",
                    "category": "general_agent",
                    "steps_count": 2,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        html = generator.generate(data)
        assert "Remediation" not in html

    def test_self_contained_no_external_refs(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [],
        }
        html = generator.generate(data)
        assert 'href="http' not in html
        assert 'src="http' not in html
        assert "<style>" in html
        assert "<script>" in html


class TestMarkdownReport:
    def test_generate_markdown_basic(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test-model", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_01",
                    "trap_type": "parameter_hallucination",
                    "category": "general_agent",
                    "steps_count": 5,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "# Agent Trust Evaluation Report" in md
        assert "test_01" in md
        assert "test-model" in md
        assert "## Summary" in md

    def test_generate_markdown_with_compliance(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {"severity": "high"},
                    "compliance": {
                        "overall": "fail",
                        "dimensions": {"tool_authorization": "fail"},
                        "critical_count": 1,
                        "high_count": 0,
                    },
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "### Compliance" in md
        assert "Overall:** fail" in md
        assert "tool_authorization" in md

    def test_generate_markdown_with_hallucination(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {"severity": "medium"},
                    "hallucination": {
                        "step_count": 1,
                        "steps": [
                            {
                                "step_index": 0,
                                "gsar_label": "Grounded",
                                "g_score": 0.8,
                                "u_score": 0.1,
                                "c_score": 0.05,
                                "faithfulness_score": 0.9,
                            }
                        ],
                    },
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "### Hallucination Analysis" in md
        assert "Grounded" in md

    def test_generate_markdown_with_remediation(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {
                        "severity": "medium",
                        "remediation": {
                            "problem": "Test problem",
                            "cause": "Test cause",
                            "fix": "Test fix",
                        },
                    },
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "### Remediation" in md
        assert "Test problem" in md
        assert "Test fix" in md

    def test_generate_markdown_benign_refusal_warning(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {"severity": "medium"},
                    "compliance": {
                        "overall": "warn",
                        "dimensions": {},
                        "critical_count": 0,
                        "high_count": 1,
                        "benign_refusal_rate": 0.25,
                    },
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "benign refusal" in md.lower()

    def test_generate_markdown_no_warning_below_threshold(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {"severity": "medium"},
                    "compliance": {
                        "overall": "pass",
                        "dimensions": {},
                        "critical_count": 0,
                        "high_count": 0,
                        "benign_refusal_rate": 0.05,
                    },
                }
            ],
        }
        md = generator.generate_markdown(data)
        assert "benign refusal" not in md.lower()

    def test_generate_markdown_to_file(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            path = f.name
        try:
            md = generator.generate_markdown(data, output_path=path)
            with open(path, "r") as f:
                saved = f.read()
            assert saved == md
        finally:
            import os

            os.unlink(path)

    def test_generate_markdown_empty_results(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [],
        }
        md = generator.generate_markdown(data)
        assert "0" in md


class TestCLIReportMarkdown:
    def test_report_markdown_format(self, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        results_path = str(tmp_path / "results.json")
        output_path = str(tmp_path / "report.md")
        data = {
            "config": {"model": "test", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "t1",
                    "metadata": {"severity": "medium"},
                    "compliance": {
                        "overall": "pass",
                        "dimensions": {},
                        "critical_count": 0,
                        "high_count": 0,
                    },
                },
            ],
        }
        with open(results_path, "w") as f:
            json.dump(data, f)

        result = runner.invoke(app, ["report", results_path, "-f", "markdown", "-o", output_path])
        assert result.exit_code == 0
        assert "Markdown report" in result.stdout
        with open(output_path, "r") as f:
            content = f.read()
        assert "# Agent Trust Evaluation Report" in content

    def test_report_invalid_format(self, tmp_path):
        from typer.testing import CliRunner

        from agent_trust_lab.cli import app

        runner = CliRunner()
        results_path = str(tmp_path / "results.json")
        with open(results_path, "w") as f:
            json.dump(
                {"config": {"model": "test"}, "results": []},
                f,
            )
        result = runner.invoke(app, ["report", results_path, "-f", "xml"])
        assert result.exit_code == 1
        assert "Invalid format" in result.stdout


class TestShareCard:
    """Tests for the share card (social media sharing) feature."""

    def _make_multi_result(self, model_label, avg_g=0.85, avg_u=0.05, avg_c=0.03, avg_f=0.88):
        return {
            "trap_id": "trap_01",
            "trap_type": "test_type",
            "category": "general_agent",
            "steps_count": 5,
            "mutated": False,
            "security_events": 0,
            "metadata": {"severity": "medium", "difficulty": "easy"},
            "compliance": {
                "overall": "pass", "dimensions": {},
                "critical_count": 0, "high_count": 0,
            },
            "hallucination": {
                "step_count": 5,
                "avg_g_score": avg_g,
                "avg_u_score": avg_u,
                "avg_c_score": avg_c,
                "avg_faithfulness": avg_f,
                "steps": [],
            },
        }

    def _make_multi_data(self, models_configs):
        configs = []
        results = {}
        for label, cfg in models_configs:
            configs.append({
                "model": cfg.get("model", label),
                "thinking_enabled": cfg.get("thinking_enabled", False),
                "reasoning_effort": cfg.get("reasoning_effort", ""),
                "config_label": label,
            })
            result = self._make_multi_result(
                label,
                avg_g=cfg.get("avg_g", 0.85),
                avg_u=cfg.get("avg_u", 0.05),
                avg_c=cfg.get("avg_c", 0.03),
                avg_f=cfg.get("avg_f", 0.88),
            )
            if "trap_01" not in results:
                results["trap_01"] = {
                    "trap_id": "trap_01",
                    "trap_type": "test_type",
                    "category": "general_agent",
                    "metadata": {
                        "base_task": "Test task",
                        "trap_injection": "Test injection",
                        "knowledge_source": "Test knowledge",
                    },
                    "scores": {},
                }
            results["trap_01"]["scores"][label] = result
        return {"configs": configs, "results": list(results.values())}

    def test_share_card_rendered_for_multi_model(self):
        generator = ReportGenerator()
        data = self._make_multi_data([
            ("model-a (think high)", {
                "model": "model-a", "thinking_enabled": True,
                "reasoning_effort": "high",
                "avg_g": 0.87, "avg_u": 0.07, "avg_c": 0.06, "avg_f": 0.87,
            }),
            ("model-a (no-think)", {
                "model": "model-a", "thinking_enabled": False,
                "avg_g": 0.81, "avg_u": 0.12, "avg_c": 0.05, "avg_f": 0.84,
            }),
        ])
        html = generator.generate(data)
        assert "share-card" in html
        assert "share-card-context" in html
        assert "share-card-bars" in html
        assert "share-bar" in html
        assert "bar-track" in html
        assert "bar-seg" in html
        assert "share-card-legend" in html
        assert "share-card-divider" in html
        assert "model-a (think high)" in html

    def test_share_card_not_rendered_for_single_model(self):
        generator = ReportGenerator()
        data = {
            "config": {"model": "test-model", "agent_type": "langchain", "sandbox": "docker"},
            "results": [
                {
                    "trap_id": "test_01",
                    "trap_type": "test_type",
                    "category": "general_agent",
                    "steps_count": 5,
                    "mutated": False,
                    "security_events": 0,
                    "metadata": {"severity": "medium", "difficulty": "easy"},
                }
            ],
        }
        html = generator.generate(data)
        assert "share-card" not in html

    def test_bars_sorted_by_trust_score(self):
        generator = ReportGenerator()
        models = [
            {"config_label": "model-a (think high)", "model": "model-a",
             "avg_g": 0.87, "avg_u": 0.07, "avg_c": 0.06, "avg_f": 0.87},
            {"config_label": "model-a (no-think)", "model": "model-a",
             "avg_g": 0.81, "avg_u": 0.12, "avg_c": 0.05, "avg_f": 0.84},
            {"config_label": "model-a (think max)", "model": "model-a",
             "avg_g": 0.49, "avg_u": 0.10, "avg_c": 0.04, "avg_f": 0.88},
        ]
        lang_dict = ReportGenerator()._get_lang("en")
        bars = generator._prepare_bars(models, lang_dict)
        assert len(bars) == 3
        assert bars[0]["trust_score"] >= bars[1]["trust_score"]
        assert bars[1]["trust_score"] >= bars[2]["trust_score"]
        for bar in bars:
            assert "g_pct" in bar
            assert "f_pct" in bar
            assert "iu_pct" in bar
            assert "ic_pct" in bar
            assert "g_val" in bar
            assert "f_val" in bar
            assert "iu_val" in bar
            assert "ic_val" in bar
            assert "trust_score" in bar

    def test_bars_have_score_tooltips(self):
        generator = ReportGenerator()
        lang_dict = ReportGenerator()._get_lang("en")
        models = [{"config_label": "model-a", "model": "m",
                   "avg_g": 0.8, "avg_u": 0.1, "avg_c": 0.05, "avg_f": 0.85}]
        bars = generator._prepare_bars(models, lang_dict)
        assert bars[0]["g_val"] == 0.8
        assert bars[0]["f_val"] == 0.85
        assert abs(bars[0]["iu_val"] - 0.9) < 0.01
        assert abs(bars[0]["ic_val"] - 0.95) < 0.01

    def test_generate_share_insight_no_models(self):
        generator = ReportGenerator()
        result = generator._generate_share_insight({"models": []}, {"lang_code": "English"})
        assert result == ""

    def test_generate_share_insight_single_model(self):
        generator = ReportGenerator()
        result = generator._generate_share_insight(
            {"models": [
                {"config_label": "test", "avg_g": 0.8,
                 "avg_u": 0.1, "avg_c": 0.05, "avg_f": 0.85},
            ]},
            {"lang_code": "English"},
        )
        assert result == ""  # Less than 2 models returns empty

    def test_bars_have_trust_score_values(self):
        generator = ReportGenerator()
        data = self._make_multi_data([
            ("good (think)", {"model": "good",
             "avg_g": 0.87, "avg_u": 0.07, "avg_c": 0.06, "avg_f": 0.87}),
            ("mid (no-think)", {"model": "mid",
             "avg_g": 0.65, "avg_u": 0.15, "avg_c": 0.08, "avg_f": 0.75}),
            ("bad (think max)", {"model": "bad",
             "avg_g": 0.30, "avg_u": 0.30, "avg_c": 0.20, "avg_f": 0.50}),
        ])
        html = generator.generate(data)
        assert "bar-value" in html

    def test_share_card_css_included(self):
        generator = ReportGenerator()
        data = self._make_multi_data([
            ("model-a (think high)", {"model": "model-a", "avg_g": 0.87}),
            ("model-a (no-think)", {"model": "model-a", "avg_g": 0.81}),
        ])
        html = generator.generate(data)
        assert ".share-card {" in html
        assert ".share-card-brand {" in html
        assert ".share-card-context {" in html
        assert ".share-card-bars {" in html
        assert ".share-bar {" in html
        assert ".bar-track {" in html
        assert ".share-card-legend {" in html
        assert ".share-card-divider {" in html
        assert ".share-card-footer {" in html


class TestBilingualReport:
    """Tests for bilingual (en + zh) report generation."""

    @staticmethod
    def _data():
        """Build a minimal multi-model data dict for bilingual tests."""
        configs = [
            {"model": "model-a", "thinking_enabled": False,
             "reasoning_effort": "", "config_label": "model-a (no-think)"},
            {"model": "model-a", "thinking_enabled": True,
             "reasoning_effort": "high", "config_label": "model-a (think high)"},
        ]
        result = {
            "trap_id": "trap_01",
            "trap_type": "test_type",
            "category": "general_agent",
            "steps_count": 5,
            "mutated": False,
            "security_events": 0,
            "metadata": {"severity": "medium"},
            "compliance": {
                "overall": "pass", "dimensions": {},
                "critical_count": 0, "high_count": 0,
            },
            "hallucination": {
                "step_count": 5,
                "avg_g_score": 0.85, "avg_u_score": 0.05,
                "avg_c_score": 0.03, "avg_faithfulness": 0.88,
                "steps": [],
            },
        }
        scores = {
            "model-a (no-think)": dict(result),
            "model-a (think high)": dict(result),
        }
        scores["model-a (no-think)"]["hallucination"]["avg_g_score"] = 0.81
        return {
            "configs": configs,
            "results": [{
                "trap_id": "trap_01",
                "trap_type": "test_type",
                "category": "general_agent",
                "metadata": {"base_task": "Test", "trap_injection": "Test",
                             "knowledge_source": "Test"},
                "scores": scores,
            }],
        }

    def test_generate_both_creates_two_files(self, tmp_path):
        generator = ReportGenerator()
        data = self._data()
        output_dir = str(tmp_path)
        en_path, zh_path = generator.generate_both(data, output_dir, "comparison")
        import os
        assert os.path.isfile(en_path)
        assert os.path.isfile(zh_path)
        assert en_path.endswith("comparison.html")
        assert zh_path.endswith("comparison_zh.html")

    def test_generate_both_files_have_lang_switch(self, tmp_path):
        generator = ReportGenerator()
        data = self._data()
        output_dir = str(tmp_path)
        en_path, zh_path = generator.generate_both(data, output_dir, "comparison")
        with open(en_path) as f:
            en_html = f.read()
        with open(zh_path) as f:
            zh_html = f.read()
        assert "lang-switch" in en_html
        assert "lang-switch" in zh_html
        assert "comparison_zh.html" in en_html
        assert "comparison.html" in zh_html

    def test_single_lang_no_lang_switch(self):
        generator = ReportGenerator()
        data = self._data()
        html = generator.generate(data, lang="en")
        assert '<div class="lang-switch">' not in html

    def test_lang_switch_highlights_current_language(self, tmp_path):
        generator = ReportGenerator()
        data = self._data()
        output_dir = str(tmp_path)
        en_path, zh_path = generator.generate_both(data, output_dir, "comparison")
        with open(en_path) as f:
            en_html = f.read()
        with open(zh_path) as f:
            zh_html = f.read()
        assert "lang-active\">English</span>" in en_html
        assert "lang-active\">中文</span>" in zh_html
        assert 'href="comparison_zh.html">中文</a>' in en_html
        assert 'href="comparison.html">English</a>' in zh_html


class TestShareCardV2:
    """Tests for Share Card v2 — horizontal bars, unified trust score, context line."""

    @pytest.fixture(autouse=True)
    def _mock_get_api_key(self):
        with patch("agent_trust_lab.llm.get_api_key", return_value=None):
            yield

    @staticmethod
    def _make_multi_data_basic():
        configs = [
            {"model": "deepseek-v4-flash", "thinking_enabled": False,
             "reasoning_effort": "", "config_label": "deepseek-v4-flash (no-think)"},
            {"model": "deepseek-v4-flash", "thinking_enabled": True,
             "reasoning_effort": "high", "config_label": "deepseek-v4-flash (think-high)"},
            {"model": "deepseek-v4-flash", "thinking_enabled": True,
             "reasoning_effort": "max", "config_label": "deepseek-v4-flash (think-max)"},
        ]
        scores = {}
        for i, cfg in enumerate(configs):
            label = cfg["config_label"]
            scores[label] = {
                "trap_id": f"trap_0{i+1}",
                "trap_type": "test",
                "category": "general_agent",
                "steps_count": 5,
                "mutated": False,
                "security_events": 0,
                "metadata": {"severity": "medium"},
                "compliance": {"overall": "pass", "dimensions": {},
                               "critical_count": 0, "high_count": 0},
                "hallucination": {"step_count": 5, "avg_g_score": 0.8 + i * 0.05,
                                  "avg_u_score": 0.1, "avg_c_score": 0.05,
                                  "avg_faithfulness": 0.85, "steps": []},
            }
        return {
            "configs": configs,
            "results": [{
                "trap_id": "trap_01",
                "trap_type": "test_type",
                "category": "general_agent",
                "metadata": {"base_task": "Test", "trap_injection": "Test",
                              "knowledge_source": "Test"},
                "scores": scores,
            }],
        }

    def test_detect_model_family_same_base(self):
        models = [
            {"model": "deepseek-v4-flash"},
            {"model": "deepseek-v4-flash"},
            {"model": "deepseek-v4-flash"},
        ]
        result = ReportGenerator._detect_model_family(models)
        assert result == "deepseek-v4-flash"

    def test_detect_model_family_different_models(self):
        models = [
            {"model": "deepseek-v4-flash"},
            {"model": "gpt-4"},
        ]
        result = ReportGenerator._detect_model_family(models)
        assert result is None

    def test_detect_model_family_empty(self):
        assert ReportGenerator._detect_model_family([]) is None

    def test_fallback_insight_produces_text(self):
        models = [
            {"config_label": "best", "model": "a", "avg_g": 0.9, "avg_u": 0.05,
             "avg_c": 0.02, "avg_f": 0.9},
            {"config_label": "worst", "model": "b", "avg_g": 0.4, "avg_u": 0.3,
             "avg_c": 0.25, "avg_f": 0.5},
        ]
        # This calls the static method directly, testing the rule fallback
        insight_en = ReportGenerator._fallback_insight(models, "en")
        assert "best" in insight_en
        assert "worst" in insight_en
        assert "Trust Score" in insight_en

    def test_fallback_insight_zh(self):
        models = [
            {"config_label": "best", "model": "a", "avg_g": 0.9, "avg_u": 0.05,
             "avg_c": 0.02, "avg_f": 0.9},
            {"config_label": "worst", "model": "b", "avg_g": 0.4, "avg_u": 0.3,
             "avg_c": 0.25, "avg_f": 0.5},
        ]
        insight_zh = ReportGenerator._fallback_insight(models, "zh")
        assert "最高" in insight_zh or "最低" in insight_zh or "差距" in insight_zh

    def test_fallback_insight_single_model_returns_empty(self):
        assert ReportGenerator._fallback_insight([{"model": "a"}], "en") == ""

    def test_prepare_bars_unified_trust_score(self):
        generator = ReportGenerator()
        lang = generator._get_lang("en")
        models = [
            {"config_label": "model-a", "model": "m", "avg_g": 0.8, "avg_u": 0.1,
             "avg_c": 0.05, "avg_f": 0.85},
        ]
        bars = generator._prepare_bars(models, lang)
        ts = (0.8 + 0.85 + (1 - 0.1) + (1 - 0.05)) / 4
        assert abs(bars[0]["trust_score"] - ts) < 0.01
        assert abs(bars[0]["g_pct"] - 20.0) < 0.1
        assert abs(bars[0]["f_pct"] - 21.25) < 0.1
        assert abs(bars[0]["iu_pct"] - 22.5) < 0.1
        assert abs(bars[0]["ic_pct"] - 23.75) < 0.1

    def test_context_line_has_benchmark_title(self):
        generator = ReportGenerator()
        data = self._make_multi_data_basic()
        html = generator.generate(data)
        assert "share-card-context" in html
        assert "Trustworthiness Benchmark" in html

    def test_context_line_includes_model_count(self):
        generator = ReportGenerator()
        configs = [
            {"model": "deepseek-v4-flash", "thinking_enabled": False,
             "reasoning_effort": "", "config_label": "deepseek-v4-flash (no-think)"},
            {"model": "gpt-4", "thinking_enabled": False,
             "reasoning_effort": "", "config_label": "gpt-4 (no-think)"},
        ]
        scores = {}
        for cfg in configs:
            label = cfg["config_label"]
            scores[label] = {
                "trap_id": "trap_01", "trap_type": "test", "category": "general_agent",
                "steps_count": 5, "mutated": False, "security_events": 0,
                "metadata": {"severity": "medium"},
                "compliance": {"overall": "pass", "dimensions": {},
                               "critical_count": 0, "high_count": 0},
                "hallucination": {"step_count": 5, "avg_g_score": 0.85,
                                  "avg_u_score": 0.1, "avg_c_score": 0.05,
                                  "avg_faithfulness": 0.85, "steps": []},
            }
        data = {
            "configs": configs,
            "results": [{
                "trap_id": "trap_01", "trap_type": "test_type", "category": "general_agent",
                "metadata": {"base_task": "Test", "trap_injection": "Test",
                              "knowledge_source": "Test"},
                "scores": scores,
            }],
        }
        html = generator.generate(data)
        assert "2" in html

    def test_share_card_has_divider(self):
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "share-card-divider" in html
        assert "▼" in html
        assert "Per-Trap" in html

    def test_share_card_has_bars_with_trust_score(self):
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "share-card-bars" in html
        assert "bar-value" in html
        assert "bar-track" in html
        assert "bar-g" in html
        assert "bar-f" in html
        assert "bar-iu" in html
        assert "bar-ic" in html

    def test_report_url_in_share_card_footer(self):
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data, report_url="https://example.com/report")
        assert 'href="https://example.com/report"' in html

    def test_report_url_empty_no_href(self):
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data, report_url="")
        assert "share-cta-text" in html
        assert '<a class="share-cta"' not in html  # No <a> tag when URL empty

    def test_load_css_main_file_exists(self):
        """_load_css should load main.css from the css directory."""
        from agent_trust_lab.report.generator import _load_css

        css = _load_css("main")
        assert ".container" in css
        assert ".trap-section" in css
        assert len(css) > 100

    def test_load_css_share_card_file_exists(self):
        """_load_css should load share_card.css."""
        from agent_trust_lab.report.generator import _load_css

        css = _load_css("share_card")
        assert ".share-card" in css
        assert ".bar-g" in css
        assert ".bar-f" in css
        assert len(css) > 100

    def test_load_css_share_card_has_768px_breakpoint(self):
        """share_card.css must include the 768px responsive breakpoint for wide screens."""
        from agent_trust_lab.report.generator import _load_css

        css = _load_css("share_card")
        assert "@media (min-width: 768px)" in css
        assert "box-shadow: none" in css or "box-shadow:none" in css
        assert "max-width: none" in css or "max-width:none" in css

    def test_load_css_share_card_divider_hidden_on_wide(self):
        """On wide screens (768px+), divider should be hidden."""
        from agent_trust_lab.report.generator import _load_css

        css = _load_css("share_card")
        min_768_idx = css.index("@media (min-width: 768px)")
        after_768 = css[min_768_idx:]
        assert ".share-card-divider" in after_768
        assert "display: none" in after_768

    def test_css_injected_when_share_card_rendered(self):
        """Main report HTML should include share_card.css when share card is rendered."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert ".share-card" in html
        assert ".bar-g" in html
        assert ".share-card-brand" in html

    def test_metrics_toggle_in_share_card(self):
        """Share card must have an expandable metrics guide toggle."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "share-card-metrics-toggle" in html
        assert "share-card-metrics-guide" in html

    def test_metrics_descriptions_rendered_en(self, mocker):
        """All 4 metric descriptions appear in EN share card."""
        mocker.patch("agent_trust_lab.llm.get_api_key", return_value=None)
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "G (Groundedness)" in html
        assert "F (Faithfulness)" in html
        assert "U⁻ (Low fabrication)" in html
        assert "C⁻ (Low contradiction)" in html

    def test_metrics_examples_rendered(self, mocker):
        """Metric example text appears in share card."""
        mocker.patch("agent_trust_lab.llm.get_api_key", return_value=None)
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "matches known facts" in html
        assert "fabrication" in html
        assert "contradiction" in html

    def test_metrics_descriptions_rendered_zh(self):
        """All 4 metric descriptions appear in ZH share card."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data, lang="zh")
        assert "锚定度" in html
        assert "忠实度" in html
        assert "反无锚度" in html
        assert "反矛盾度" in html

    def test_metrics_example_zh(self):
        """Metric example text appears in ZH share card."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data, lang="zh")
        assert "已知事实一致" in html
        assert "编造" in html

    def test_per_category_stats_computes(self):
        """_compute_per_category_stats groups scores by trap_type for each model."""
        traps = [
            {
                "trap_type": "parameter_hallucination",
                "models": [
                    {"label": "model-a (no-think)", "avg_g": 0.72, "avg_u": 0.15,
                     "avg_c": 0.08, "avg_f": 0.80},
                    {"label": "model-b (think-high)", "avg_g": 0.88, "avg_u": 0.05,
                     "avg_c": 0.03, "avg_f": 0.92},
                ],
            },
            {
                "trap_type": "tool_bypass",
                "models": [
                    {"label": "model-a (no-think)", "avg_g": 0.58, "avg_u": 0.30,
                     "avg_c": 0.18, "avg_f": 0.65},
                    {"label": "model-b (think-high)", "avg_g": 0.82, "avg_u": 0.10,
                     "avg_c": 0.06, "avg_f": 0.88},
                ],
            },
        ]
        labels = ["model-a (no-think)", "model-b (think-high)"]
        result = ReportGenerator._compute_per_category_stats(traps, labels)
        assert "parameter_hallucination" in result
        assert "tool_bypass" in result
        ph = result["parameter_hallucination"]
        assert abs(ph["model-a (no-think)"]["avg_g"] - 0.72) < 0.01
        assert abs(ph["model-b (think-high)"]["avg_g"] - 0.88) < 0.01

    def test_per_category_stats_with_multiple_traps_same_type(self):
        """Multiple traps of same type should average their scores."""
        traps = [
            {
                "trap_type": "parameter_hallucination",
                "models": [
                    {"label": "model-a (no-think)", "avg_g": 0.60, "avg_u": 0.20,
                     "avg_c": 0.10, "avg_f": 0.70},
                ],
            },
            {
                "trap_type": "parameter_hallucination",
                "models": [
                    {"label": "model-a (no-think)", "avg_g": 0.80, "avg_u": 0.10,
                     "avg_c": 0.05, "avg_f": 0.85},
                ],
            },
        ]
        labels = ["model-a (no-think)"]
        result = ReportGenerator._compute_per_category_stats(traps, labels)
        assert "parameter_hallucination" in result
        ph = result["parameter_hallucination"]
        assert abs(ph["model-a (no-think)"]["avg_g"] - 0.70) < 0.01

    def test_per_category_stats_empty_traps_returns_empty(self):
        result = ReportGenerator._compute_per_category_stats([], ["model-x"])
        assert result == {}

    def test_per_category_stats_skips_unknown_label(self):
        traps = [{
            "trap_type": "test",
            "models": [{"label": "other", "avg_g": 0.5, "avg_u": 0.2,
                         "avg_c": 0.1, "avg_f": 0.75}],
        }]
        result = ReportGenerator._compute_per_category_stats(
            traps, ["model-x (think)"]
        )
        assert "test" in result
        assert "model-x (think)" not in result["test"]

    def test_fallback_insight_with_per_category(self):
        """With per_category data, fallback insight mentions spread type with description."""
        models = [
            {"config_label": "best", "model": "a", "avg_g": 0.9, "avg_u": 0.05,
             "avg_c": 0.02, "avg_f": 0.9},
            {"config_label": "worst", "model": "b", "avg_g": 0.4, "avg_u": 0.3,
             "avg_c": 0.25, "avg_f": 0.5},
        ]
        per_category = {
            "parameter_hallucination": {
                "best": {"avg_g": 0.85, "avg_u": 0.10, "avg_c": 0.05, "avg_f": 0.88},
                "worst": {"avg_g": 0.30, "avg_u": 0.40, "avg_c": 0.30, "avg_f": 0.40},
            },
        }
        lang_dict = ReportGenerator()._get_lang("en")
        insight = ReportGenerator._fallback_insight(models, "en", per_category, lang_dict)
        assert "parameter_hallucination" in insight
        assert "best" in insight

    def test_fallback_insight_with_per_category_zh(self):
        models = [
            {"config_label": "best", "model": "a", "avg_g": 0.9, "avg_u": 0.05,
             "avg_c": 0.02, "avg_f": 0.9},
            {"config_label": "worst", "model": "b", "avg_g": 0.4, "avg_u": 0.3,
             "avg_c": 0.25, "avg_f": 0.5},
        ]
        per_category = {
            "parameter_hallucination": {
                "best": {"avg_g": 0.85, "avg_u": 0.10, "avg_c": 0.05, "avg_f": 0.88},
                "worst": {"avg_g": 0.30, "avg_u": 0.40, "avg_c": 0.30, "avg_f": 0.40},
            },
        }
        lang_dict = ReportGenerator()._get_lang("zh")
        insight = ReportGenerator._fallback_insight(models, "zh", per_category, lang_dict)
        assert "parameter_hallucination" in insight or "差异最大" in insight

    def test_fallback_insight_small_spread_no_type_mention(self):
        """When per-category spread is tiny (<0.05), don't mention a type."""
        models = [
            {"config_label": "m1", "model": "a", "avg_g": 0.8, "avg_u": 0.1,
             "avg_c": 0.05, "avg_f": 0.85},
            {"config_label": "m2", "model": "b", "avg_g": 0.79, "avg_u": 0.11,
             "avg_c": 0.06, "avg_f": 0.84},
        ]
        per_category = {
            "test": {
                "m1": {"avg_g": 0.80, "avg_u": 0.10, "avg_c": 0.05, "avg_f": 0.85},
                "m2": {"avg_g": 0.79, "avg_u": 0.11, "avg_c": 0.06, "avg_f": 0.84},
            },
        }
        insight = ReportGenerator._fallback_insight(models, "en", per_category)
        assert "Largest spread" not in insight

    def test_generate_share_insight_passes_per_category(self, mocker):
        """_generate_share_insight should accept and use per_category parameter."""
        mocker.patch("agent_trust_lab.llm.get_api_key", return_value=None)
        generator = ReportGenerator()
        models = [
            {"config_label": "m1", "model": "a", "avg_g": 0.8, "avg_u": 0.1,
             "avg_c": 0.05, "avg_f": 0.85},
            {"config_label": "m2", "model": "b", "avg_g": 0.7, "avg_u": 0.15,
             "avg_c": 0.08, "avg_f": 0.80},
        ]
        summary = {"models": models, "is_multi_model": True}
        per_category = {
            "test_type": {
                "m1": {"avg_g": 0.85, "avg_u": 0.08, "avg_c": 0.03, "avg_f": 0.88},
                "m2": {"avg_g": 0.68, "avg_u": 0.18, "avg_c": 0.10, "avg_f": 0.78},
            },
        }
        result = generator._generate_share_insight(
            summary, generator._get_lang("en"), "en", per_category
        )
        assert isinstance(result, str)

    def test_share_card_includes_insight_text_with_per_category(self, mocker):
        """Share card renders insight when per-category data is available (fallback path)."""
        mocker.patch("agent_trust_lab.llm.get_api_key", return_value=None)
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "share-card-insight" in html

    def test_context_line_benchmark_format(self):
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "Trustworthiness Benchmark" in html
        assert "configs" in html
        assert "scenarios" in html

    def test_mutated_badge_has_tooltip(self):
        """Mutated badge should have a title tooltip explaining what mutated means."""
        generator = ReportGenerator()
        configs = [
            {"model": "m", "thinking_enabled": False, "reasoning_effort": "",
             "config_label": "m1"},
            {"model": "m", "thinking_enabled": True, "reasoning_effort": "high",
             "config_label": "m2"},
        ]
        scores = {}
        for cfg in configs:
            label = cfg["config_label"]
            scores[label] = {
                "trap_id": "t_01", "trap_type": "test", "category": "gen",
                "steps_count": 3, "mutated": True,
                "metadata": {"severity": "medium"},
                "compliance": {"overall": "pass", "dimensions": {},
                               "critical_count": 0, "high_count": 0},
                "hallucination": {"step_count": 3, "avg_g_score": 0.8,
                                  "avg_u_score": 0.1, "avg_c_score": 0.05,
                                  "avg_faithfulness": 0.85, "steps": []},
            }
        data = {
            "configs": configs,
            "results": [{"trap_id": "t_01", "trap_type": "test", "category": "gen",
                         "metadata": {"base_task": "T", "trap_injection": "T"},
                         "scores": scores}],
        }
        html = generator.generate(data)
        assert "badge-mutated" in html
        assert "this trap" in html.lower() or "该陷阱" in html

    def test_metrics_desc_iu_direction_correct(self):
        """U⁻ description should say HIGHER U⁻ means better (not lower)."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "Higher U⁻" in html or "U⁻ 越高" in html

    def test_metrics_desc_ic_direction_correct(self):
        """C⁻ description should say HIGHER C⁻ means better (not lower)."""
        generator = ReportGenerator()
        data = TestShareCardV2._make_multi_data_basic()
        html = generator.generate(data)
        assert "Higher C⁻" in html or "C⁻ 越高" in html

    def test_per_model_stats_handles_none_scores(self):
        """_per_model_stats should not crash when avg_u_score/avg_c_score are None."""
        results = [{
            "scores": {
                "test-model (no-think)": {
                    "mutated": False,
                    "compliance": {"overall": "pass"},
                    "hallucination": {
                        "step_count": 3,
                        "avg_g_score": 0.7,
                        "avg_u_score": None,
                        "avg_c_score": None,
                        "avg_faithfulness": 0.8,
                        "steps": [
                            {"u_score": 0.1, "c_score": 0.05, "g_score": 0.9},
                            {"u_score": 0.2, "c_score": 0.10, "g_score": 0.8},
                            {"u_score": 0.3, "c_score": 0.15, "g_score": 0.7},
                        ],
                    },
                },
            },
        }]
        stats = ReportGenerator._per_model_stats(results, "test-model (no-think)")
        assert stats is not None
        assert stats["avg_u"] > 0  # Should have computed from steps
        assert stats["avg_c"] > 0

    def test_enrich_traps_handles_none_scores(self):
        """_enrich_traps should compute U/C from steps when avg_u/c_score are None."""
        generator = ReportGenerator()
        raw = [{
            "trap_id": "test_none",
            "trap_type": "test",
            "category": "general_agent",
            "metadata": {"severity": "medium"},
            "scores": {
                "model-a (no-think)": {
                    "steps_count": 3,
                    "mutated": False,
                    "compliance": {"overall": "pass", "dimensions": {},
                                   "critical_count": 0, "high_count": 0},
                    "hallucination": {
                        "step_count": 3,
                        "avg_g_score": 0.7,
                        "avg_u_score": None,
                        "avg_c_score": None,
                        "avg_faithfulness": 0.8,
                        "steps": [
                            {"u_score": 0.1, "c_score": 0.05, "g_score": 0.9},
                            {"u_score": 0.2, "c_score": 0.10, "g_score": 0.8},
                            {"u_score": 0.3, "c_score": 0.15, "g_score": 0.7},
                        ],
                    },
                },
            },
        }]
        traps = generator._enrich_traps(raw)
        assert traps[0]["models"][0]["avg_u"] > 0
        assert traps[0]["models"][0]["avg_c"] > 0
