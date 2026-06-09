"""Report generation facade. Delegates to format-specific sub-generators."""

from typing import Any, Dict, List, Optional

from agent_trust_lab.report._shared import (
    _compute_per_category_stats,
    _compute_summary,
    _detect_model_family,
    _enrich_traps,
    _fallback_insight,
    _generate_share_insight,
    _get_lang,
    _load_css,
    _per_model_stats,
    _prepare_bars,
    I18N,
)
from agent_trust_lab.report.comparison import ComparisonReportGenerator
from agent_trust_lab.report.html_report import HtmlReportGenerator
from agent_trust_lab.report.markdown_report import MarkdownReportGenerator


class ReportGenerator:
    """Report generation facade. Delegates to format-specific generators."""

    def generate(self, data, output_path=None, calibration=None, lang="en", lang_other_url="", report_url=""):
        gen = HtmlReportGenerator()
        return gen.generate(data, output_path, calibration, lang, lang_other_url=lang_other_url, report_url=report_url)

    def generate_markdown(self, data, output_path=None, calibration=None, lang="en"):
        gen = MarkdownReportGenerator()
        return gen.generate(data, output_path, calibration, lang)

    def generate_both(self, data, output_dir, base_name, calibration=None, report_url=""):
        gen = HtmlReportGenerator()
        return gen.generate_both(data, output_dir, base_name, calibration, report_url)

    @staticmethod
    def merge_results(json_paths):
        return ComparisonReportGenerator.merge_results(json_paths)

    @classmethod
    def from_json_file(cls, json_path: str) -> str:
        """Convenience: load a JSON export file and generate HTML report."""
        return HtmlReportGenerator.from_json_file(json_path)

    # -- Backward-compatible delegation to _shared internals (used by tests) --

    def _get_lang(self, lang: str = "en") -> Dict[str, str]:
        return _get_lang(lang)

    @staticmethod
    def _detect_model_family(models: List[Dict[str, Any]]) -> Optional[str]:
        return _detect_model_family(models)

    @staticmethod
    def _fallback_insight(
        models: List[Dict[str, Any]],
        lang_code: str,
        per_category: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        lang_dict: Optional[Dict[str, str]] = None,
    ) -> str:
        return _fallback_insight(models, lang_code, per_category, lang_dict)

    @staticmethod
    def _compute_per_category_stats(
        traps: List[Dict[str, Any]],
        model_labels: List[str],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        return _compute_per_category_stats(traps, model_labels)

    @staticmethod
    def _per_model_stats(
        raw_results: List[Dict[str, Any]],
        model_label: str,
        difficulty_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict]:
        return _per_model_stats(raw_results, model_label, difficulty_weights)

    @staticmethod
    def _prepare_bars(
        models: List[Dict[str, Any]], lang_dict: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        return _prepare_bars(models, lang_dict)

    @staticmethod
    def _enrich_traps(
        raw_results: List[Dict[str, Any]],
        calibration: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return _enrich_traps(raw_results, calibration)

    def _generate_share_insight(
        self,
        summary: Dict[str, Any],
        lang_dict: Dict[str, str],
        lang_code: str = "en",
        per_category: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> str:
        return _generate_share_insight(summary, lang_dict, lang_code, per_category)
