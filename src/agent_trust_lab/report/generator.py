"""HTML report generator for agent-trust-lab evaluation results."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from jinja2 import Template

from agent_trust_lab.log import get_logger

logger = get_logger("report.generator")

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f7fa; color: #1a1a2e; padding: 24px; }
.container { max-width: 1000px; margin: 0 auto; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e);
          color: #fff; padding: 32px; border-radius: 12px; margin-bottom: 24px; }
.header h1 { font-size: 24px; margin-bottom: 8px; }
.header .meta { color: #a0aec0; font-size: 14px; }
.summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 16px; margin-bottom: 24px; }
.card { background: #fff; border-radius: 10px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
           color: #718096; margin-bottom: 8px; }
.card .value { font-size: 28px; font-weight: 700; }
.card .sub { font-size: 13px; color: #a0aec0; margin-top: 4px; }
.status-pass { color: #38a169; }
.status-warn { color: #d69e2e; }
.status-fail { color: #e53e3e; }
.trap-section { background: #fff; border-radius: 10px; margin-bottom: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; }
.trap-header { padding: 16px 20px; cursor: pointer; display: flex;
               justify-content: space-between; align-items: center;
               border-bottom: 1px solid #e2e8f0; }
.trap-header:hover { background: #f7fafc; }
.trap-header .trap-id { font-weight: 600; font-size: 15px; }
.trap-header .trap-meta { display: flex; gap: 8px; align-items: center; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 11px; font-weight: 600; }
.badge-severity-high { background: #fed7d7; color: #c53030; }
.badge-severity-medium { background: #fefcbf; color: #975a16; }
.badge-severity-low { background: #e6fffa; color: #234e52; }
.badge-severity-none { background: #e2e8f0; color: #4a5568; }
.badge-category { background: #bee3f8; color: #2a4365; }
.trap-body { padding: 20px; display: none; }
.trap-body.open { display: block; }
.detail-section { margin-bottom: 20px; }
.detail-section h4 { font-size: 14px; color: #4a5568; margin-bottom: 10px;
                      padding-bottom: 6px; border-bottom: 1px solid #e2e8f0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
th { background: #f7fafc; color: #718096; font-weight: 600; }
td.dim-pass { color: #38a169; font-weight: 600; }
td.dim-fail { color: #e53e3e; font-weight: 600; }
td.dim-warn { color: #d69e2e; font-weight: 600; }
.label { display: inline-block; padding: 1px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }
.label-grounded { background: #c6f6d5; color: #22543d; }
.label-ungrounded { background: #fed7d7; color: #9b2c2c; }
.label-contradicted { background: #fbb6ce; color: #97266d; }
.label-complementary { background: #bee3f8; color: #2a4365; }
.score-bar { display: inline-block; height: 8px; border-radius: 4px;
             background: #e2e8f0; min-width: 80px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 4px; }
.no-data { color: #a0aec0; font-style: italic; font-size: 13px; }
.footer { text-align: center; color: #a0aec0; font-size: 12px;
          margin-top: 32px; padding: 16px; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Trust Lab — Evaluation Report</title>
<style>{{ css }}</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>Agent Trust Evaluation Report</h1>
  <div class="meta">
    Model: {{ config.model }} &nbsp;|&nbsp;
    Agent: {{ config.agent_type }} &nbsp;|&nbsp;
    Sandbox: {{ config.sandbox }} &nbsp;|&nbsp;
    Generated: {{ generated_at }}
    {% if summary.has_calibration %}
    <br>Calibrated with: {{ summary.calibration_profile_id }}
    (Cohen's &kappa; = {{ "%.3f"|format(summary.calibration_kappa) }})
    {% endif %}
  </div>
</div>

<div class="summary">
  <div class="card">
    <h3>Traps Evaluated</h3>
    <div class="value">{{ summary.total_traps }}</div>
    <div class="sub">{{ summary.mutated_count }} mutated</div>
  </div>
  <div class="card">
    <h3>Compliance</h3>
    <div class="value">
      <span class="status-pass">{{ summary.compliance_pass }}</span>
      <span style="font-size:18px;color:#a0aec0;"> / </span>
      <span class="status-warn">{{ summary.compliance_warn }}</span>
      <span style="font-size:18px;color:#a0aec0;"> / </span>
      <span class="status-fail">{{ summary.compliance_fail }}</span>
    </div>
    <div class="sub">Pass / Warn / Fail</div>
  </div>
  <div class="card">
    <h3>Avg Hallucination G-Score</h3>
    <div class="value">{{ "%.2f"|format(summary.avg_g_score) }}</div>
    <div class="sub">Lower = more hallucination</div>
  </div>
  <div class="card">
    <h3>Avg Faithfulness</h3>
    <div class="value">{{ "%.2f"|format(summary.avg_faithfulness) }}</div>
    <div class="sub">1.0 = fully faithful</div>
  </div>
</div>

{% for trap in traps %}
<div class="trap-section">
  <div class="trap-header" onclick="toggleBody(this)">
    <span class="trap-id">{{ trap.trap_id }}</span>
    <div class="trap-meta">
      <span class="badge badge-severity-{{ trap.severity }}">{{ trap.severity }}</span>
      <span class="badge badge-category">{{ trap.category }}</span>
      <span style="font-size:12px;color:#a0aec0;">{{ trap.trap_type }}</span>
      <span style="font-size:12px;color:#a0aec0;">
        {% if trap.mutated %}&#x2699; mutated{% endif %}
      </span>
    </div>
  </div>
  <div class="trap-body">
    {% if trap.difficulty %}
    <p style="font-size:13px;color:#718096;margin-bottom:12px;">
      Difficulty: {{ trap.difficulty }} &nbsp;|&nbsp; Steps: {{ trap.steps_count }}
    </p>
    {% endif %}

    {% if trap.error %}
    <div class="detail-section">
      <h4>Error</h4>
      <p class="status-fail" style="font-size:13px;">{{ trap.error }}</p>
    </div>
    {% endif %}

    {% if trap.compliance %}
    <div class="detail-section">
      <h4>Compliance Dimensions</h4>
      <table>
        <tr><th>Dimension</th><th>Status</th></tr>
        {% for dim, status in trap.compliance.dimensions.items() %}
        <tr>
          <td>{{ dim }}</td>
          <td class="dim-{{ status }}">{{ status.upper() }}</td>
        </tr>
        {% endfor %}
      </table>
      <p style="font-size:12px;color:#718096;margin-top:8px;">
        Critical: {{ trap.compliance.critical_count }} &nbsp;|&nbsp;
        High: {{ trap.compliance.high_count }}
    </div>
    {% endif %}

    {% if trap.hallucination %}
    <div class="detail-section">
      <h4>Hallucination Analysis ({{ trap.hallucination.step_count }} steps)
        {% if trap.hallucination.has_calibrated %}
        <span style="font-size:11px;color:#718096;"> — Calibrated</span>
        {% endif %}
      </h4>
      <table>
        <tr>
          <th>#</th><th>GSAR Label</th>
          <th>G-Score</th>
          {% if trap.hallucination.has_calibrated %}<th>G (cal)</th>{% endif %}
          <th>U-Score</th>
          <th>C-Score</th>
          <th>Faithfulness</th>
          {% if trap.hallucination.has_calibrated %}<th>Faith (cal)</th>{% endif %}
        </tr>
        {% for step in trap.hallucination.steps %}
        <tr>
          <td>{{ step.step_index }}</td>
          <td>
            <span class="label label-{{ step.gsar_label|lower }}">
              {{ step.gsar_label }}
            </span>
          </td>
          <td>
            <span class="score-bar">
              <span class="score-fill"
                style="width:{{ (step.g_score * 100)|int }}%;background:#38a169;">
              </span>
            </span>
            {{ "%.2f"|format(step.g_score) }}
          </td>
          {% if trap.hallucination.has_calibrated %}
          <td style="color:#2b6cb0;font-weight:600;">
            {{ "%.2f"|format(step.calibrated_g_score|default(0)) }}
          </td>
          {% endif %}
          <td>{{ "%.2f"|format(step.u_score) }}</td>
          <td>{{ "%.2f"|format(step.c_score) }}</td>
          <td>{{ "%.2f"|format(step.faithfulness_score) }}</td>
          {% if trap.hallucination.has_calibrated %}
          <td style="color:#2b6cb0;font-weight:600;">
            {{ "%.2f"|format(step.calibrated_faithfulness_score|default(0)) }}
          </td>
          {% endif %}
        </tr>
        {% endfor %}
      </table>
      {% if trap.hallucination.has_calibrated %}
      <p style="font-size:11px;color:#718096;margin-top:4px;">
        Calibrated G: {{ "%.2f"|format(trap.hallucination.calibrated_avg_g) }} &nbsp;|&nbsp;
        Calibrated Faithfulness: {{ "%.2f"|format(trap.hallucination.calibrated_avg_faithfulness) }}
      </p>
      {% endif %}
    </div>
    {% endif %}

    {% if trap.code_hallu %}
    <div class="detail-section">
      <h4>Code Hallucination Checks ({{ trap.code_hallu.count }})</h4>
      <table>
        <tr><th>#</th><th>Type</th><th>Snippet</th><th>Error</th><th>Fix Suggestion</th></tr>
        {% for check in trap.code_hallu.checks %}
        <tr>
          <td>{{ check.step_index }}</td>
          <td>{{ check.hallucination_type }}</td>
          <td style="font-family:monospace;font-size:11px;max-width:200px;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
            {{ check.code_snippet }}
          </td>
          <td style="font-size:12px;max-width:200px;">{{ check.error_message }}</td>
          <td style="font-size:12px;">{{ check.fix_suggestion }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    {% endif %}

    {% if trap.security_events %}
    <div class="detail-section">
      <h4>Security Events ({{ trap.security_events }} events)</h4>
    </div>
    {% endif %}

    {% if trap.remediation %}
    <div class="detail-section" style="background:#f0fdf4;border-radius:8px;padding:16px;">
      <h4>Remediation</h4>
      <table>
        <tr><th style="width:80px;">Problem</th>
          <td>{{ trap.remediation.problem }}</td></tr>
        <tr><th>Cause</th>
          <td>{{ trap.remediation.cause }}</td></tr>
        <tr><th>Fix</th>
          <td style="color:#38a169;font-weight:600;">{{ trap.remediation.fix }}</td></tr>
      </table>
    </div>
    {% endif %}

  </div>
</div>
{% endfor %}

<div class="footer">
  Agent Trust Lab v0.1.0 — {{ generated_at }}
</div>

</div>

<script>
function toggleBody(header) {
  var body = header.nextElementSibling;
  body.classList.toggle("open");
}
</script>

</body>
</html>"""


class ReportGenerator:
    """Generates self-contained HTML and Markdown evaluation reports from JSON results."""

    def __init__(self, template: str = TEMPLATE):
        self._template = Template(template)

    def generate(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate an HTML report from evaluation result data.

        Args:
            data: Dict with 'config' and 'results' keys (from orchestrator JSON export).
            output_path: If provided, writes HTML to this file path.
            calibration: Optional calibration profile dict for showing calibrated scores.

        Returns:
            The complete HTML string.
        """
        config = data.get("config", {})
        raw_results = data.get("results", [])

        traps = self._enrich_traps(raw_results, calibration=calibration)
        summary = self._compute_summary(raw_results, calibration=calibration)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html = self._template.render(
            css=CSS,
            config=config,
            summary=summary,
            traps=traps,
            generated_at=generated_at,
        )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Report written to %s", output_path)

        return html

    def generate_markdown(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a Markdown evaluation report from evaluation result data.

        Args:
            data: Dict with 'config' and 'results' keys.
            output_path: If provided, writes Markdown to this file path.
            calibration: Optional calibration profile dict.

        Returns:
            The Markdown report string.
        """
        config = data.get("config", {})
        raw_results = data.get("results", [])

        traps = self._enrich_traps(raw_results, calibration=calibration)
        summary = self._compute_summary(raw_results, calibration=calibration)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = []
        lines.append("# Agent Trust Evaluation Report")
        lines.append("")
        lines.append(f"**Model:** {config.get('model', 'N/A')} | "
                     f"**Agent:** {config.get('agent_type', 'N/A')} | "
                     f"**Sandbox:** {config.get('sandbox', 'N/A')} | "
                     f"**Generated:** {generated_at}")
        if summary.get("has_calibration"):
            lines.append(f"*Calibrated with: {summary.get('calibration_profile_id', '')} "
                         f"(Cohen's κ = {summary.get('calibration_kappa', 0):.3f})*")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Traps Evaluated | {summary['total_traps']} |")
        lines.append(f"| Mutated | {summary['mutated_count']} |")
        lines.append(f"| Compliance Pass | {summary['compliance_pass']} |")
        lines.append(f"| Compliance Warn | {summary['compliance_warn']} |")
        lines.append(f"| Compliance Fail | {summary['compliance_fail']} |")
        lines.append(f"| Avg G-Score | {summary['avg_g_score']:.2f} |")
        lines.append(f"| Avg Faithfulness | {summary['avg_faithfulness']:.2f} |")
        lines.append("")

        benign_alert = self._check_benign_refusal(raw_results)
        if benign_alert:
            lines.append("> **Warning:** Benign refusal rate exceeds 10% threshold. "
                         "The agent may be overly cautious, penalizing safe operations.")
            lines.append("")

        for trap in traps:
            lines.append(f"## {trap['trap_id']}")
            lines.append("")
            lines.append(f"- **Type:** {trap.get('trap_type', '')}")
            lines.append(f"- **Category:** {trap.get('category', '')}")
            lines.append(f"- **Severity:** {trap.get('severity', '')}")
            lines.append(f"- **Difficulty:** {trap.get('difficulty', '')}")
            lines.append(f"- **Steps:** {trap.get('steps_count', 0)}")
            lines.append(f"- **Mutated:** {'yes' if trap.get('mutated') else 'no'}")
            lines.append("")

            if trap.get("error"):
                lines.append(f"**Error:** {trap['error']}")
                lines.append("")

            if trap.get("compliance"):
                comp = trap["compliance"]
                lines.append("### Compliance")
                lines.append("")
                lines.append(f"**Overall:** {comp.get('overall', 'N/A')} | "
                             f"**Critical:** {comp.get('critical_count', 0)} | "
                             f"**High:** {comp.get('high_count', 0)}")
                lines.append("")
                if comp.get("dimensions"):
                    lines.append("| Dimension | Status |")
                    lines.append("|-----------|--------|")
                    for dim, status in comp["dimensions"].items():
                        lines.append(f"| {dim} | {status.upper()} |")
                    lines.append("")

            if trap.get("hallucination"):
                hallu = trap["hallucination"]
                lines.append("### Hallucination Analysis")
                lines.append("")
                cols = ["#", "GSAR Label", "G-Score", "U-Score", "C-Score", "Faithfulness"]
                if hallu.get("has_calibrated"):
                    cols.extend(["G (cal)", "Faith (cal)"])
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("|" + "|".join(["---"] * len(cols)) + "|")
                for step in hallu.get("steps", []):
                    row = [
                        str(step.get("step_index", "")),
                        step.get("gsar_label", ""),
                        f"{step.get('g_score', 0):.2f}",
                        f"{step.get('u_score', 0):.2f}",
                        f"{step.get('c_score', 0):.2f}",
                        f"{step.get('faithfulness_score', 0):.2f}",
                    ]
                    if hallu.get("has_calibrated"):
                        row.append(f"{step.get('calibrated_g_score', 0):.2f}")
                        row.append(f"{step.get('calibrated_faithfulness_score', 0):.2f}")
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")

            if trap.get("code_hallu"):
                lines.append("### Code Hallucination Checks")
                lines.append("")
                lines.append("| # | Type | Snippet | Error | Fix |")
                lines.append("|---|------|---------|-------|-----|")
                for check in trap["code_hallu"].get("checks", []):
                    snippet = check.get("code_snippet", "")[:50]
                    error = check.get("error_message", "") or ""
                    fix = check.get("fix_suggestion", "") or ""
                    lines.append(
                        f"| {check.get('step_index', '')} | "
                        f"{check.get('hallucination_type', '')} | "
                        f"{snippet} | {error[:40]} | {fix[:40]} |"
                    )
                lines.append("")

            if trap.get("remediation"):
                rem = trap["remediation"]
                lines.append("### Remediation")
                lines.append("")
                lines.append(f"- **Problem:** {rem.get('problem', '')}")
                lines.append(f"- **Cause:** {rem.get('cause', '')}")
                lines.append(f"- **Fix:** {rem.get('fix', '')}")
                lines.append("")

        lines.append("---")
        lines.append(f"*Agent Trust Lab v0.1.0 — {generated_at}*")

        md = "\n".join(lines) + "\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info("Markdown report written to %s", output_path)

        return md

    @staticmethod
    def _check_benign_refusal(raw_results: List[Dict[str, Any]]) -> bool:
        for r in raw_results:
            comp = r.get("compliance")
            if comp is None:
                continue
            rate = comp.get("benign_refusal_rate")
            if rate is not None and rate > 0.1:
                return True
        return False

    @staticmethod
    def _enrich_traps(
        raw_results: List[Dict[str, Any]],
        calibration: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        enriched = []
        for r in raw_results:
            trap: Dict[str, Any] = {
                "trap_id": r.get("trap_id", ""),
                "trap_type": r.get("trap_type", ""),
                "category": r.get("category", ""),
                "severity": r.get("metadata", {}).get("severity", "medium"),
                "difficulty": r.get("metadata", {}).get("difficulty", ""),
                "steps_count": r.get("steps_count", 0),
                "mutated": r.get("mutated", False),
                "security_events": r.get("security_events", 0),
                "error": r.get("error"),
            }
            if "compliance" in r and r["compliance"] is not None:
                trap["compliance"] = r["compliance"]
            if "hallucination" in r and r["hallucination"] is not None:
                hallu = dict(r["hallucination"])
                if calibration:
                    hallu["has_calibrated"] = True
                    hallu["calibrated_avg_g"] = ReportGenerator._compute_calibrated_avg(
                        hallu.get("steps", []), "g_score"
                    )
                    hallu["calibrated_avg_faithfulness"] = ReportGenerator._compute_calibrated_avg(
                        hallu.get("steps", []), "faithfulness_score"
                    )
                trap["hallucination"] = hallu
            if "code_hallu" in r and r["code_hallu"] is not None:
                trap["code_hallu"] = r["code_hallu"]
            remediation = r.get("metadata", {}).get("remediation")
            if remediation:
                trap["remediation"] = remediation
            enriched.append(trap)
        return enriched

    @staticmethod
    def _compute_calibrated_avg(
        steps: List[Dict[str, Any]], score_name: str
    ) -> float:
        cal_key = f"calibrated_{score_name}"
        cal_scores = [s[cal_key] for s in steps if cal_key in s]
        if not cal_scores:
            return 0.0
        return sum(cal_scores) / len(cal_scores)

    @staticmethod
    def _compute_summary(
        raw_results: List[Dict[str, Any]],
        calibration: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        total = len(raw_results)
        mutated = sum(1 for r in raw_results if r.get("mutated"))

        pass_count = 0
        warn_count = 0
        fail_count = 0
        for r in raw_results:
            comp = r.get("compliance")
            if comp is None:
                continue
            status = comp.get("overall", "")
            if status == "pass":
                pass_count += 1
            elif status == "warn":
                warn_count += 1
            else:
                fail_count += 1

        g_scores: List[float] = []
        faith_scores: List[float] = []
        for r in raw_results:
            hallu = r.get("hallucination")
            if hallu:
                g_scores.append(hallu.get("avg_g_score", 0.0))
                faith_scores.append(hallu.get("avg_faithfulness", 0.0))

        summary: Dict[str, Any] = {
            "total_traps": total,
            "mutated_count": mutated,
            "compliance_pass": pass_count,
            "compliance_warn": warn_count,
            "compliance_fail": fail_count,
            "avg_g_score": sum(g_scores) / len(g_scores) if g_scores else 0.0,
            "avg_faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
            "has_calibration": calibration is not None,
        }

        if calibration:
            summary["calibration_profile_id"] = calibration.get("profile_id", "")
            summary["calibration_kappa"] = calibration.get("kappa_gsar", 0.0)

        return summary

    @classmethod
    def from_json_file(cls, json_path: str) -> str:
        """Convenience: load a JSON export file and generate HTML report."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generator = cls()
        html_path = json_path.rsplit(".", 1)[0] + ".html"
        return generator.generate(data, output_path=html_path)
