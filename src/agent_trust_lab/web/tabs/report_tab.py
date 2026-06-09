"""Tab 2: Report Viewer — upload results JSON and inspect contents."""

import json

import gradio as gr

from agent_trust_lab.web.i18n import t as _t


def build_report_tab(_i18n_outputs: list, **kwargs) -> None:
    """Build the Report Viewer tab inside the current Gradio context.

    Args:
        _i18n_outputs: Shared list — append every Gradio component here
            in creation order so the language-switch callback can update them.
    """
    report_load_md = gr.Markdown(_t("md_load_results", "en"))
    _i18n_outputs.append(report_load_md)

    report_json = gr.File(
        label=_t("label_upload_results", "en"),
        file_types=[".json"],
    )
    _i18n_outputs.append(report_json)

    report_content = gr.JSON(label=_t("label_report_content", "en"))
    _i18n_outputs.append(report_content)

    def on_report_upload(file):
        if file is None:
            return None
        with open(file.name, "r", encoding="utf-8") as f:
            return json.load(f)

    report_json.change(
        fn=on_report_upload,
        inputs=[report_json],
        outputs=[report_content],
    )
