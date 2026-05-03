"""Tests for report generator."""

import json
import tempfile
from pathlib import Path

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
        assert "share-card-champion" in html
        assert "share-card-radar" in html
        assert "share-card-metrics" in html
        assert "svg" in html
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

    def test_radar_svg_generated(self):
        generator = ReportGenerator()
        models = [
            {"config_label": "model-a (think high)", "model": "model-a",
             "avg_g": 0.87, "avg_u": 0.07, "avg_c": 0.06, "avg_f": 0.87},
            {"config_label": "model-a (no-think)", "model": "model-a",
             "avg_g": 0.81, "avg_u": 0.12, "avg_c": 0.05, "avg_f": 0.84},
            {"config_label": "model-a (think max)", "model": "model-a",
             "avg_g": 0.49, "avg_u": 0.10, "avg_c": 0.04, "avg_f": 0.88},
        ]
        svg = generator._render_radar_svg(models, max_polygons=5)
        assert "<svg" in svg
        assert "</svg>" in svg
        assert 'aria-label="Model comparison radar chart"' in svg
        assert "polygon" in svg
        assert "G" in svg
        assert "1-U" in svg
        assert "1-C" in svg
        assert "circle" in svg

    def test_radar_svg_limits_polygons(self):
        generator = ReportGenerator()
        models = []
        for i in range(10):
            models.append({
                "config_label": f"model-{i}",
                "model": "test",
                "avg_g": 0.8 - i * 0.05,
                "avg_u": 0.1,
                "avg_c": 0.05,
                "avg_f": 0.85,
            })
        svg = generator._render_radar_svg(models, max_polygons=3)
        polygon_count = svg.count('<polygon points=')
        grid_count = 5  # 5 grid level polygons
        assert polygon_count == grid_count + 3  # grid + 3 model polygons

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

    def test_ranking_score_based_stars(self):
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
        assert "★★★★" in html  # good model
        assert "★★★" in html  # mid model (0.6-0.8)
        assert "★" in html   # bad model (<0.4), but ★ is a subset of ★★/★★★, test more carefully
        # Verify the correct number of stars for each rank
        # rank 1 (0.87): ★★★★
        # rank 2 (0.65): ★★★
        # rank 3 (0.30): ★
        assert "★★★★" in html
        assert "★★★" in html
        assert '<td class="rank-stars">★</td>' in html or '<td class="rank-stars">★' in html

    def test_share_card_css_included(self):
        generator = ReportGenerator()
        data = self._make_multi_data([
            ("model-a (think high)", {"model": "model-a", "avg_g": 0.87}),
            ("model-a (no-think)", {"model": "model-a", "avg_g": 0.81}),
        ])
        html = generator.generate(data)
        assert ".share-card {" in html
        assert ".share-card-brand {" in html
        assert ".share-card-champion {" in html
        assert ".share-card-radar {" in html
        assert ".share-card-insight {" in html
        assert ".share-card-metrics {" in html
        assert ".share-card-ranking {" in html
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
