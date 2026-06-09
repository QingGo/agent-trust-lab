"""Tab 1: Run Evaluation — trap selection, agent config, execution, and results."""

import json
import os
import tempfile
from typing import Dict, List

import gradio as gr

from agent_trust_lab.web._shared import (
    _build_trap_choices,
    _get_trap_info,
    _run_evaluation,
)
from agent_trust_lab.web.i18n import t as _t


def build_run_tab(
    trap_choices: Dict[str, List[str]],
    all_trap_ids: List[str],
    categories: List[str],
    _i18n_outputs: list,
    **kwargs,
) -> None:
    """Build the Run Evaluation tab inside the current Gradio context.

    Args:
        trap_choices: Map of category -> list of trap IDs.
        all_trap_ids: Flattened, sorted, deduplicated list of all trap IDs.
        categories: Sorted list of trap category names.
        _i18n_outputs: Shared list — append every Gradio component here
            in creation order so the language-switch callback can update them.
    """
    with gr.Row():
        with gr.Column(scale=1):
            trap_sel_md = gr.Markdown(_t("md_trap_selection", "en"))
            _i18n_outputs.append(trap_sel_md)

            trap_category = gr.Dropdown(
                label=_t("label_category", "en"),
                choices=sorted(categories),
                value=categories[0] if categories else None,
            )
            _i18n_outputs.append(trap_category)

            trap_id_dropdown = gr.Dropdown(
                label=_t("label_trap_id", "en"),
                choices=all_trap_ids,
                info=_t("info_select_trap", "en"),
            )
            _i18n_outputs.append(trap_id_dropdown)

            trap_details_accordion = gr.Accordion(
                _t("accordion_trap_details", "en"), open=False
            )
            _i18n_outputs.append(trap_details_accordion)
            with trap_details_accordion:
                trap_info = gr.JSON(label=_t("label_trap_info", "en"))
                _i18n_outputs.append(trap_info)

        with gr.Column(scale=1):
            agent_cfg_md = gr.Markdown(_t("md_agent_config", "en"))
            _i18n_outputs.append(agent_cfg_md)

            model_input = gr.Textbox(
                label=_t("label_model", "en"), value="deepseek-v4-flash"
            )
            _i18n_outputs.append(model_input)

            agent_type = gr.Dropdown(
                label=_t("label_harness", "en"),
                choices=[
                    "langchain",
                    "codex",
                    "openai",
                    "opencode",
                    "claude-code",
                    "gemini-cli",
                ],
                value="langchain",
            )
            _i18n_outputs.append(agent_type)

            sandbox_input = gr.Dropdown(
                label=_t("label_sandbox", "en"),
                choices=["docker", "dry-run"],
                value="docker",
            )
            _i18n_outputs.append(sandbox_input)

            thinking_checkbox = gr.Checkbox(
                label=_t("label_thinking", "en"), value=False
            )
            _i18n_outputs.append(thinking_checkbox)

            effort_dropdown = gr.Dropdown(
                label=_t("label_effort", "en"),
                choices=["", "high", "max"],
                value="",
                visible=False,
            )
            _i18n_outputs.append(effort_dropdown)

            mutate_checkbox = gr.Checkbox(
                label=_t("label_mutate", "en"), value=False
            )
            _i18n_outputs.append(mutate_checkbox)

    thinking_checkbox.change(
        fn=lambda v: gr.Dropdown(visible=v),
        inputs=[thinking_checkbox],
        outputs=[effort_dropdown],
    )

    def on_trap_category_change(cat):
        ids = sorted(set(trap_choices.get(cat, [])))
        return gr.Dropdown(choices=ids, value=ids[0] if ids else None)

    trap_category.change(
        fn=on_trap_category_change,
        inputs=[trap_category],
        outputs=[trap_id_dropdown],
    )

    def on_trap_select(trap_id):
        if not trap_id:
            return None
        return _get_trap_info(trap_id)

    trap_id_dropdown.change(
        fn=on_trap_select,
        inputs=[trap_id_dropdown],
        outputs=[trap_info],
    )

    run_button = gr.Button(_t("btn_run", "en"), variant="primary", size="lg")
    _i18n_outputs.append(run_button)

    results_accordion = gr.Accordion(_t("accordion_results", "en"), open=True)
    _i18n_outputs.append(results_accordion)
    with results_accordion:
        compliance_box = gr.Textbox(label=_t("label_compliance", "en"), lines=3)
        _i18n_outputs.append(compliance_box)

        hallucination_box = gr.JSON(label=_t("label_hallu", "en"))
        _i18n_outputs.append(hallucination_box)

        steps_box = gr.Markdown(
            label=_t("label_steps_md", "en"), elem_classes=["step-log"]
        )
        _i18n_outputs.append(steps_box)

        json_download = gr.File(label=_t("label_download_result", "en"))
        _i18n_outputs.append(json_download)

    def on_run(trap_id, model, agent_type_val, sandbox, thinking, effort, mutate):
        full_agent_type = agent_type_val if agent_type_val != "docker" else "langchain"
        result = _run_evaluation(
            trap_id,
            model,
            full_agent_type,
            sandbox,
            thinking,
            effort,
            mutate,
        )
        compliance = "No compliance data"
        if "compliance" in result:
            comp = result["compliance"]
            dims = "\n".join(
                f"- {k}: {v}" for k, v in comp.get("dimensions", {}).items()
            )
            compliance = (
                f"Overall: {comp.get('overall', 'N/A').upper()}\n"
                f"Critical: {comp.get('critical_count', 0)}, "
                f"High: {comp.get('high_count', 0)}\n"
                f"Dimensions:\n{dims}"
            )

        hallu = result.get("hallucination", {})
        hallu_data = {
            "Avg G-Score": round(hallu.get("avg_g_score", 0), 4),
            "Avg U-Score": round(hallu.get("avg_u_score", 0), 4),
            "Avg C-Score": round(hallu.get("avg_c_score", 0), 4),
            "Avg Faithfulness": round(hallu.get("avg_faithfulness", 0), 4),
            "Labels": hallu.get("labels", []),
            "Step Count": hallu.get("step_count", 0),
        }

        steps = hallu.get("steps", [])
        steps_md = ""
        for s in steps:
            atype = s.get("anchor_type", "none")
            steps_md += (
                f"**Step {s['step_index']}** | "
                f"Type: `{s.get('step_type', '?')}` | "
                f"GSAR: `{s['gsar_label']}` | "
                f"Faith: `{s['faithfulness_score']:.3f}` | "
                f"Anchor: `{atype}`\n\n"
                f"```\n{s.get('step_content', '(no content)')[:300]}\n```\n"
                f"---\n"
            )
        if not steps_md:
            steps_md = "No hallucination steps generated"

        tmpdir = tempfile.mkdtemp()
        json_path = os.path.join(tmpdir, f"{trap_id}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return compliance, hallu_data, steps_md, json_path

    run_button.click(
        fn=on_run,
        inputs=[
            trap_id_dropdown,
            model_input,
            agent_type,
            sandbox_input,
            thinking_checkbox,
            effort_dropdown,
            mutate_checkbox,
        ],
        outputs=[compliance_box, hallucination_box, steps_box, json_download],
    )
