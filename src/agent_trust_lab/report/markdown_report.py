"""Markdown report generator for agent-trust-lab evaluation results."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_trust_lab.log import get_logger
from agent_trust_lab.report._shared import (
    _check_benign_refusal,
    _compute_summary,
    _enrich_traps,
    _get_lang,
)

logger = get_logger("report.markdown_report")


class MarkdownReportGenerator:
    """Generates Markdown evaluation reports from JSON results."""

    def generate(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
        lang: str = "en",
    ) -> str:
        """Generate a Markdown evaluation report from evaluation result data.

        Args:
            data: Dict with 'config' and 'results' keys.
            output_path: If provided, writes Markdown to this file path.
            calibration: Optional calibration profile dict.
            lang: Language code (en/zh).

        Returns:
            The Markdown report string.
        """
        config = data.get("config", {})
        raw_results = data.get("results", [])
        lang_dict = _get_lang(lang)

        traps = _enrich_traps(raw_results, calibration=calibration)
        configs = data.get("configs", None)
        if configs:
            diff_weights = configs[0].get("difficulty_weights", {}) if configs else {}
        else:
            diff_weights = config.get("difficulty_weights", {})
        summary = _compute_summary(raw_results, calibration=calibration, data=data,
                                   difficulty_weights=diff_weights)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = []
        lines.append(f"# {lang_dict['title']}")
        lines.append("")
        lines.append(
            f"**Model:** {config.get('model', 'N/A')} | "
            f"**Agent:** {config.get('agent_type', 'N/A')} | "
            f"**Sandbox:** {config.get('sandbox', 'N/A')} | "
            f"**Generated:** {generated_at}"
        )
        if summary.get("has_calibration"):
            lines.append(
                f"*Calibrated: {summary.get('calibration_profile_id', '')} "
                f"(κ = {summary.get('calibration_kappa', 0):.3f})*"
            )
        lines.append("")

        lines.append(f"## {lang_dict['summary_title']}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| {lang_dict['traps_evaluated']} | {summary['total_traps']} |")
        lines.append(f"| {lang_dict['mutated_count']} | {summary['mutated_count']} |")
        lines.append(f"| {lang_dict['compliance_header']} Pass | {summary['compliance_pass']} |")
        lines.append(f"| {lang_dict['compliance_header']} Warn | {summary['compliance_warn']} |")
        lines.append(f"| {lang_dict['compliance_header']} Fail | {summary['compliance_fail']} |")
        lines.append(f"| {lang_dict['avg_g_score']} | {summary['avg_g_score']:.2f} |")
        lines.append(f"| {lang_dict['avg_faithfulness']} | {summary['avg_faithfulness']:.2f} |")
        lines.append("")

        if summary.get("benign_refusal_warning"):
            lines.append(f"> **{lang_dict['benign_warning']}**")
            lines.append("")

        benign_alert = _check_benign_refusal(raw_results)
        if benign_alert:
            lines.append(
                "> **Warning:** Benign refusal rate exceeds 10% threshold. "
                "The agent may be overly cautious, penalizing safe operations."
            )
            lines.append("")

        for trap in traps:
            lines.append(f"## {trap['trap_id']}")
            lines.append("")
            lines.append(
                f"- **{lang_dict['trap_context']}:** "
                f"{trap.get('trap_type', '')} | "
                f"{trap.get('category', '')} | "
                f"{trap.get('severity', '')}"
            )
            lines.append(
                f"- **Steps:** {trap.get('steps_count', 0)} | "
                f"**Mutated:** {'yes' if trap.get('mutated') else 'no'}"
            )
            lines.append("")

            if trap.get("trap_context"):
                ctx = trap["trap_context"]
                if ctx.get("base_task"):
                    lines.append(f"**{lang_dict['trap_purpose']}:** {ctx['base_task']}")
                if ctx.get("trap_injection"):
                    lines.append(f"**{lang_dict['trap_injection']}:** {ctx['trap_injection']}")
                if ctx.get("knowledge_source"):
                    lines.append(
                        f"**{lang_dict['trap_knowledge']}:** {ctx['knowledge_source'][:120]}"
                    )
                lines.append("")

            if trap.get("error"):
                lines.append(f"**Error:** {trap['error']}")
                lines.append("")

            if trap.get("compliance"):
                comp = trap["compliance"]
                lines.append(f"### {lang_dict['compliance_title']}")
                lines.append("")
                lines.append(
                    f"**{lang_dict['overall']}:** {comp.get('overall', 'N/A')} | "
                    f"**{lang_dict['critical']}:** {comp.get('critical_count', 0)} | "
                    f"**{lang_dict['high']}:** {comp.get('high_count', 0)}"
                )
                lines.append("")
                if comp.get("dimensions"):
                    lines.append("| Dimension | Status |")
                    lines.append("|-----------|--------|")
                    for dim, status in comp["dimensions"].items():
                        lines.append(f"| {dim} | {status.upper()} |")
                    lines.append("")

            if trap.get("hallucination"):
                hallu = trap["hallucination"]
                lines.append(
                    f"### {lang_dict['hallu_title'].replace('{n}', str(hallu.get('step_count', 0)))}"
                )
                lines.append("")
                cols = [
                    "#",
                    lang_dict["table_label"],
                    lang_dict["table_g"],
                    lang_dict["table_u"],
                    lang_dict["table_c"],
                    lang_dict["table_f"],
                ]
                if hallu.get("has_calibrated"):
                    cols.extend(["G (cal)", "F (cal)"])
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
                lines.append(
                    f"### {lang_dict['code_hallu_title'].replace('{n}', str(trap['code_hallu'].get('count', 0)))}"
                )
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
                lines.append(f"### {lang_dict['remediation_title']}")
                lines.append("")
                lines.append(f"- **{lang_dict['remediation_problem']}:** {rem.get('problem', '')}")
                lines.append(f"- **{lang_dict['remediation_cause']}:** {rem.get('cause', '')}")
                lines.append(f"- **{lang_dict['remediation_fix']}:** {rem.get('fix', '')}")
                lines.append("")

        lines.append("---")
        lines.append(f"*{lang_dict['footer']} — {generated_at}*")

        md = "\n".join(lines) + "\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info("Markdown report written to %s", output_path)

        return md
