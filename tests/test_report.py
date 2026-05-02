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
