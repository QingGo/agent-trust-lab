import gradio as gr

from agent_trust_lab.web._shared import (
    UI_CSS,
    _build_trap_choices,
)
from agent_trust_lab.web.i18n import t as _t
from agent_trust_lab.web.tabs import (
    build_about_tab,
    build_annotation_tab,
    build_report_tab,
    build_run_tab,
)


def create_ui() -> gr.Blocks:
    """Create the full Gradio UI by composing tab builders.

    This function is intentionally thin — it pre-computes shared data
    (trap choices, statistics), then delegates to per-tab builder
    functions that each own their component creation and event wiring.
    """
    trap_choices = _build_trap_choices()
    all_trap_ids: list = []
    for ids in trap_choices.values():
        all_trap_ids.extend(ids)
    all_trap_ids = sorted(set(all_trap_ids))

    categories = sorted(trap_choices.keys())

    about_trailer = (
        f"\n\n**Available traps:** {len(all_trap_ids)} "
        f"({len(trap_choices)} categories)\n\n"
        "**8 registered harnesses:** langchain, codex, openai, opencode, "
        "claude-code, gemini-cli, docker, dry-run\n\n"
        "**2 report formats:** HTML, Markdown (en/zh)"
    )

    _i18n_outputs: list = []  # collected in creation order for on_lang_change

    with gr.Blocks(title="Agent Trust Lab") as demo:

        lang_radio = gr.Radio(
            choices=[("English", "en"), ("\u4e2d\u6587", "zh")],
            value="en",
            label=_t("lang_label", "en"),
            interactive=True,
            elem_id="lang-selector",
        )
        _i18n_outputs.append(lang_radio)

        header_md = gr.Markdown(_t("md_title", "en"))
        _i18n_outputs.append(header_md)

        with gr.Tabs():
            tab1 = gr.TabItem(_t("tab_run", "en"))
            tab2 = gr.TabItem(_t("tab_report", "en"))
            tab3 = gr.TabItem(_t("tab_about", "en"))
            tab4 = gr.TabItem(_t("tab_annotate", "en"))

        _i18n_outputs.extend([tab1, tab2, tab3, tab4])

        # ------------------------------------------------------------------
        # Tab 1: Run Evaluation
        # ------------------------------------------------------------------
        with tab1:
            build_run_tab(trap_choices, all_trap_ids, categories, _i18n_outputs)

        # ------------------------------------------------------------------
        # Tab 2: Report Viewer
        # ------------------------------------------------------------------
        with tab2:
            build_report_tab(_i18n_outputs)

        # ------------------------------------------------------------------
        # Tab 3: About
        # ------------------------------------------------------------------
        with tab3:
            build_about_tab(_i18n_outputs, about_trailer)

        # ------------------------------------------------------------------
        # Tab 4: Annotation
        # ------------------------------------------------------------------
        with tab4:
            build_annotation_tab(_i18n_outputs)

        # ------------------------------------------------------------------
        # Language switcher callback
        # ------------------------------------------------------------------
        def on_lang_change(lang: str) -> list:
            n_outputs = len(_i18n_outputs)
            return [
                # 0: lang_radio label
                gr.update(label=_t("lang_label", lang)),
                # 1: header markdown
                gr.update(value=_t("md_title", lang)),
                # 2-5: tab labels
                gr.update(label=_t("tab_run", lang)),
                gr.update(label=_t("tab_report", lang)),
                gr.update(label=_t("tab_about", lang)),
                gr.update(label=_t("tab_annotate", lang)),
                # 6-9: Run Evaluation markdown headers + dropdowns
                gr.update(value=_t("md_trap_selection", lang)),
                gr.update(label=_t("label_category", lang)),
                gr.update(
                    label=_t("label_trap_id", lang),
                    info=_t("info_select_trap", lang),
                ),
                # 10-11: accordion + json
                gr.update(label=_t("accordion_trap_details", lang)),
                gr.update(label=_t("label_trap_info", lang)),
                # 12-18: Agent config
                gr.update(value=_t("md_agent_config", lang)),
                gr.update(label=_t("label_model", lang)),
                gr.update(label=_t("label_harness", lang)),
                gr.update(label=_t("label_sandbox", lang)),
                gr.update(label=_t("label_thinking", lang)),
                gr.update(label=_t("label_effort", lang)),
                gr.update(label=_t("label_mutate", lang)),
                # 19-24: run button + results
                gr.update(value=_t("btn_run", lang)),
                gr.update(label=_t("accordion_results", lang)),
                gr.update(label=_t("label_compliance", lang)),
                gr.update(label=_t("label_hallu", lang)),
                gr.update(label=_t("label_steps_md", lang)),
                gr.update(label=_t("label_download_result", lang)),
                # 25-27: Report Viewer
                gr.update(value=_t("md_load_results", lang)),
                gr.update(label=_t("label_upload_results", lang)),
                gr.update(label=_t("label_report_content", lang)),
                # 28: About
                gr.update(value=_t("md_about", lang) + about_trailer),
                # 29-30: Annotation guide accordion + markdown
                gr.update(label=_t("accordion_guide", lang)),
                gr.update(value=_t("md_annotation_guide", lang)),
                # 31-58: Annotation tab (rest)
                gr.update(value=_t("md_load_candidates", lang)),
                gr.update(label=_t("label_upload_candidates", lang)),
                gr.update(label=_t("label_output_path", lang)),
                gr.update(label=_t("label_status", lang)),
                gr.update(value=_t("md_navigation", lang)),
                gr.update(value=_t("btn_previous", lang)),
                gr.update(value=_t("btn_next", lang)),
                gr.update(value=_t("btn_skip", lang)),
                gr.update(label=_t("label_progress", lang)),
                gr.update(value=_t("md_step_content", lang)),
                gr.update(label=_t("label_step_content", lang)),
                gr.update(label=_t("label_metadata", lang)),
                gr.update(value=_t("md_evidence", lang)),
                gr.update(label=_t("label_evidence", lang)),
                gr.update(label=_t("label_explanation", lang)),
                gr.update(value=_t("md_annotation", lang)),
                gr.update(label=_t("label_gsar", lang)),
                gr.update(label=_t("label_original", lang)),
                gr.update(
                    label=_t("label_g_score", lang),
                    info=_t("info_g_score", lang),
                ),
                gr.update(
                    label=_t("label_u_score", lang),
                    info=_t("info_u_score", lang),
                ),
                gr.update(
                    label=_t("label_c_score", lang),
                    info=_t("info_c_score", lang),
                ),
                gr.update(
                    label=_t("label_f_score", lang),
                    info=_t("info_f_score", lang),
                ),
                gr.update(value=_t("btn_submit", lang)),
                gr.update(label=_t("accordion_export", lang)),
                gr.update(value=_t("btn_export", lang)),
                gr.update(label=_t("label_download_ann", lang)),
                gr.update(label=_t("label_ann_stats", lang)),
            ][:n_outputs]

        lang_radio.change(
            fn=on_lang_change,
            inputs=[lang_radio],
            outputs=_i18n_outputs,
        )

    return demo


def launch_ui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web UI server."""
    demo = create_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme="soft",
        css=UI_CSS,
    )


__all__ = ["create_ui", "launch_ui"]
