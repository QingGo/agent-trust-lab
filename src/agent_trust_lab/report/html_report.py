"""HTML report generator for agent-trust-lab evaluation results."""

import json
import os as _os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from agent_trust_lab.log import get_logger
from agent_trust_lab.report._shared import (
    _compute_per_category_stats,
    _compute_summary,
    _enrich_traps,
    _fallback_insight,
    _generate_share_insight,
    _get_lang,
    _load_css,
    _prepare_bars,
)

logger = get_logger("report.html_report")

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    """Load a Jinja2 template from the templates directory."""
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HtmlReportGenerator
# ---------------------------------------------------------------------------

class HtmlReportGenerator:
    """Generates self-contained HTML evaluation reports from JSON results."""

    def __init__(self, template: str | None = None):
        if template is None:
            template = _load_template("report.html.jinja2")
        self._template = Template(template)

    def _render_legend(self, lang_dict: Dict[str, str], summary: Dict[str, Any]) -> str:
        t = Template(_load_template("legend.html.jinja2"))
        return t.render(lang=lang_dict, summary=summary)

    @staticmethod
    def _render_lang_switch(
        lang_dict: Dict[str, str], lang_code: str, lang_other_url: str
    ) -> str:
        """Render a language switch bar linking to the other language version."""
        if not lang_other_url:
            return ""
        other_code = "zh" if lang_code == "en" else "en"
        other_label = lang_dict.get(f"lang_switch_{other_code}", other_code)
        current_label = lang_dict.get(f"lang_switch_{lang_code}", lang_code)
        label = lang_dict.get("lang_switch_label", "Language")
        return (
            f'<div class="lang-switch">'
            f'<span style="color:#718096;font-size:12px;">{label}:</span>'
            f'<span class="lang-active">{current_label}</span>'
            f'<span class="lang-sep">|</span>'
            f'<a href="{lang_other_url}">{other_label}</a>'
            f"</div>"
        )

    def _render_share_card(
        self,
        summary: Dict[str, Any],
        lang_dict: Dict[str, str],
        generated_at: str,
        total_traps: int,
        traps: Optional[List[Dict[str, Any]]] = None,
        report_url: str = "",
        lang_code: str = "en",
    ) -> str:
        """Render the share card HTML block for social media sharing (v2: horizontal bars)."""
        if not summary.get("is_multi_model"):
            return ""
        models = summary.get("models", [])
        if len(models) < 2:
            return ""

        model_labels = [m.get("config_label", m.get("model", "")) for m in models]

        generated_date = generated_at.split(" ")[0] if " " in generated_at else generated_at
        context_line = (
            f"{lang_dict['share_card_context_title']} — "
            f"{len(models)} {lang_dict['share_card_context_configs_short']} × "
            f"{total_traps} {lang_dict['share_card_context_scenarios_short']}"
            f" · {lang_dict['share_card_context_generated']} {generated_date}"
        )

        bars = _prepare_bars(models, lang_dict)
        per_category = _compute_per_category_stats(traps or [], model_labels)
        insight_text = _generate_share_insight(
            summary, lang_dict, lang_code, per_category
        )

        t = Template(_load_template("share_card.html.jinja2"))
        return t.render(
            lang=lang_dict,
            models=models,
            total_traps=total_traps,
            bars=bars,
            context_line=context_line,
            insight_text=insight_text,
            generated_at=generated_at,
            report_url=report_url,
        )

    def generate(
        self,
        data: Dict[str, Any],
        output_path: Optional[str] = None,
        calibration: Optional[Dict[str, Any]] = None,
        lang: str = "en",
        lang_other_url: str = "",
        report_url: str = "",
    ) -> str:
        """Generate an HTML report from evaluation result data.

        Args:
            data: Dict with 'config' and 'results' keys (from orchestrator JSON export).
            output_path: If provided, writes HTML to this file path.
            calibration: Optional calibration profile dict for showing calibrated scores.
            lang: Language code (en/zh).
            lang_other_url: Optional URL to the other language version for the lang switch.
            report_url: Optional URL for the "Full report" link in the share card footer.

        Returns:
            The complete HTML string.
        """
        config = data.get("config", {})
        configs = data.get("configs", None)
        raw_results = data.get("results", [])
        lang_dict = _get_lang(lang)

        traps = _enrich_traps(raw_results, calibration=calibration)
        if configs:
            diff_weights = configs[0].get("difficulty_weights", {}) if configs else {}
        else:
            diff_weights = config.get("difficulty_weights", {})
        summary = _compute_summary(raw_results, calibration=calibration, data=data,
                                   difficulty_weights=diff_weights)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        share_card_html = self._render_share_card(
            summary, lang_dict, generated_at, summary.get("total_traps", 0),
            traps=traps, report_url=report_url, lang_code=lang,
        )

        lang_switch_html = self._render_lang_switch(lang_dict, lang, lang_other_url)

        css = _load_css("main")
        if share_card_html:
            css += "\n" + _load_css("share_card")
        html = self._template.render(
            css=css,
            lang=lang_dict,
            lang_code=lang,
            config=config,
            configs=configs,
            summary=summary,
            traps=traps,
            generated_at=generated_at,
            legend_html=self._render_legend(lang_dict, summary),
            share_card_html=share_card_html,
            lang_switch_html=lang_switch_html,
        )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Report written to %s", output_path)

        return html

    def generate_both(
        self,
        data: Dict[str, Any],
        output_dir: str,
        base_name: str = "comparison",
        calibration: Optional[Dict[str, Any]] = None,
        report_url: str = "",
    ) -> tuple:
        """Generate both English and Chinese HTML reports with cross-references.

        Args:
            data: Merged evaluation data dict.
            output_dir: Directory for output files.
            base_name: Base filename (e.g. "comparison" → "comparison.html", "comparison_zh.html").
            calibration: Optional calibration profile dict.
            report_url: Optional URL for the "Full report" link in the share card footer.

        Returns:
            Tuple of (en_path, zh_path).
        """
        en_path = _os.path.join(output_dir, f"{base_name}.html")
        zh_path = _os.path.join(output_dir, f"{base_name}_zh.html")
        en_basename = _os.path.basename(en_path)
        zh_basename = _os.path.basename(zh_path)

        self.generate(
            data,
            output_path=en_path,
            calibration=calibration,
            lang="en",
            lang_other_url=zh_basename,
            report_url=report_url,
        )
        self.generate(
            data,
            output_path=zh_path,
            calibration=calibration,
            lang="zh",
            lang_other_url=en_basename,
            report_url=report_url,
        )
        logger.info("Bilingual reports: %s, %s", en_path, zh_path)
        return en_path, zh_path

    @classmethod
    def from_json_file(cls, json_path: str) -> str:
        """Convenience: load a JSON export file and generate HTML report."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generator = cls()
        html_path = json_path.rsplit(".", 1)[0] + ".html"
        return generator.generate(data, output_path=html_path)
