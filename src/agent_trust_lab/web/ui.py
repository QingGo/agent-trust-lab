import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from agent_trust_lab.web.i18n import t as _t

GSAR_LABEL_OPTIONS = ["Grounded", "Ungrounded", "Contradicted", "Complementary"]

_LABEL_SCORE_PRESETS = {
    "Grounded": (0.90, 0.05, 0.05, 0.95),
    "Ungrounded": (0.10, 0.85, 0.10, 0.15),
    "Contradicted": (0.05, 0.10, 0.90, 0.10),
    "Complementary": (0.80, 0.10, 0.05, 0.90),
}

_SLIDER_CONSTRAINTS: dict = {
    "Grounded": (
        dict(minimum=0.50, maximum=1.0),   dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),    dict(minimum=0.50, maximum=1.0),
    ),
    "Ungrounded": (
        dict(minimum=0.0, maximum=0.49),    dict(minimum=0.50, maximum=1.0),
        dict(minimum=0.0, maximum=0.49),    dict(minimum=0.0, maximum=0.49),
    ),
    "Contradicted": (
        dict(minimum=0.0, maximum=0.49),    dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.50, maximum=1.0),    dict(minimum=0.0, maximum=0.49),
    ),
    "Complementary": (
        dict(minimum=0.50, maximum=1.0),   dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),    dict(minimum=0.50, maximum=1.0),
    ),
    "none": (
        dict(minimum=0.0, maximum=1.0),    dict(minimum=0.0, maximum=1.0),
        dict(minimum=0.0, maximum=1.0),     dict(minimum=0.0, maximum=1.0),
    ),
}


def _slider_updates(label: str | None, values: tuple) -> list:
    """Build gr.update() list for 4 sliders with label-based constraints."""
    if label and label in _SLIDER_CONSTRAINTS:
        constraints = _SLIDER_CONSTRAINTS[label]
    else:
        constraints = _SLIDER_CONSTRAINTS["none"]
    return [
        gr.update(value=values[0], **constraints[0]),
        gr.update(value=values[1], **constraints[1]),
        gr.update(value=values[2], **constraints[2]),
        gr.update(value=values[3], **constraints[3]),
    ]


def _save_annotations_web(output_path: str, annotations: list) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = {"benchmark": "gsar-calibration-v1", "version": "1.0", "annotations": annotations}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def _render_candidate(state: dict, lang: str = "en") -> tuple:
    candidates = state.get("candidates", [])
    current = state.get("current_index", 0)
    total = len(candidates)

    if not candidates or current >= total:
        msg = _t("end_of_candidates", lang).format(total)
        return _empty_step_results(state, msg)

    c = candidates[current]
    key = (c.get("trap_id"), c.get("step_index"))
    annotated_ids = state.get("annotated_ids", set())

    step = c.get("step_content", "")[:3000]
    meta_txt = (
        f"Trap: {c.get('trap_id','?')} | "
        f"Type: {c.get('trap_type','?')} | "
        f"Step: {c.get('step_index','?')} ({c.get('step_type','?')})"
    )
    ev = "\n".join(
        e[:500] if isinstance(e, str) else str(e)[:500]
        for e in (c.get("evidence", []) or [])
    ) or _t("no_evidence", lang)
    expl = (c.get("explanation", "") or "")[:800]
    orig_label = c.get("original_gsar_label", "?")

    is_annotated = key in annotated_ids
    status_text = f"{'[ANNOTATED]' if is_annotated else ''} {current + 1} / {total}"

    if is_annotated:
        for a in state.get("annotations", []):
            if a.get("trap_id") == key[0] and a.get("step_index") == key[1]:
                label = a.get("gsar_label")
                default_scores = (
                    a.get("g_score", 0.5), a.get("u_score", 0.1),
                    a.get("c_score", 0.05), a.get("faithfulness_score", 0.9),
                )
                preset = _LABEL_SCORE_PRESETS.get(label, default_scores)
                slider_vals = _slider_updates(label, preset)
                return (
                    state, status_text,
                    step, meta_txt, ev, expl, orig_label,
                    label,
                    slider_vals[0], slider_vals[1],
                    slider_vals[2], slider_vals[3],
                    status_text,
                )

    ps = _LABEL_SCORE_PRESETS.get(
        orig_label if orig_label in _LABEL_SCORE_PRESETS else "Grounded",
        (0.5, 0.1, 0.05, 0.9),
    )
    displayed_label = orig_label if orig_label in GSAR_LABEL_OPTIONS else None
    slider_vals = _slider_updates(
        orig_label if orig_label in _LABEL_SCORE_PRESETS else None, ps
    )
    return (
        state, status_text,
        step, meta_txt, ev, expl, orig_label,
        displayed_label,
        slider_vals[0], slider_vals[1],
        slider_vals[2], slider_vals[3],
        status_text,
    )


def _empty_step_results(state: dict, message: str = "No data") -> tuple:
    total = len(state.get("candidates", []))
    progress = f"{total} / {total}" if total else "0 / 0"
    default = (0.5, 0.1, 0.05, 0.9)
    slider_vals = _slider_updates(None, default)
    return (
        state, message,
        "", "", "", "", "",
        None,
        slider_vals[0], slider_vals[1],
        slider_vals[2], slider_vals[3],
        progress,
    )


def _get_traps_data_dir() -> Path:
    import agent_trust_lab.traps

    pkg_dir = Path(agent_trust_lab.traps.__file__).parent
    data_dir = pkg_dir / "data"
    if data_dir.is_dir():
        return data_dir
    import importlib.resources

    return Path(str(importlib.resources.files("agent_trust_lab.traps"))) / "data"


def _get_trap_manager():
    from agent_trust_lab.traps.manager import TrapManager

    return TrapManager(str(_get_traps_data_dir()))


def _build_trap_choices() -> Dict[str, List[str]]:
    mgr = _get_trap_manager()
    traps = mgr.load_traps(include_controls=True)
    by_category: Dict[str, List[str]] = {}
    for trap in traps:
        by_category.setdefault(trap.category, []).append(trap.trap_id)
    return by_category


def _get_trap_info(trap_id: str) -> Optional[Dict[str, Any]]:
    mgr = _get_trap_manager()
    trap = mgr.get_trap(trap_id)
    if trap is None:
        return None
    return {
        "trap_id": trap.trap_id,
        "trap_type": trap.trap_type,
        "severity": trap.severity,
        "difficulty": trap.difficulty,
        "category": trap.category,
        "base_task": trap.base_task,
        "injection": trap.trap_injection or "",
        "tools": [t.get("name", str(t)) for t in trap.tools] if trap.tools else [],
        "expected_violation": trap.expected_violation_signature or "",
    }


def _run_evaluation(
    trap_id: str,
    model: str,
    agent_type: str,
    sandbox: str,
    thinking: bool,
    effort: str,
    mutate: bool,
) -> Dict[str, Any]:
    from agent_trust_lab.config import EvaluationConfig
    from agent_trust_lab.orchestrator import Orchestrator

    config = EvaluationConfig(
        trap_library_path=str(_get_traps_data_dir()),
        agent_type=agent_type,
        model=model,
        sandbox=sandbox,
        thinking_enabled=thinking,
        reasoning_effort=effort if thinking else "",
    )

    orchestrator = Orchestrator(config)
    results = orchestrator.run_traps(trap_ids=[trap_id], mutate=mutate)
    return results[0].summary() if results else {"error": "No results"}


def create_ui() -> gr.Blocks:
    trap_choices = _build_trap_choices()
    all_trap_ids = []
    for ids in trap_choices.values():
        all_trap_ids.extend(ids)
    all_trap_ids = sorted(set(all_trap_ids))

    categories = sorted(trap_choices.keys())

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

        about_trailer = (
            f"\n\n**Available traps:** {len(all_trap_ids)} "
            f"({len(trap_choices)} categories)\n\n"
            "**8 registered harnesses:** langchain, codex, openai, opencode, "
            "claude-code, gemini-cli, docker, dry-run\n\n"
            "**2 report formats:** HTML, Markdown (en/zh)"
        )

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
                        choices=["langchain", "codex", "openai", "opencode",
                                 "claude-code", "gemini-cli"],
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
                compliance_box = gr.Textbox(
                    label=_t("label_compliance", "en"), lines=3
                )
                _i18n_outputs.append(compliance_box)

                hallucination_box = gr.JSON(label=_t("label_hallu", "en"))
                _i18n_outputs.append(hallucination_box)

                steps_box = gr.Markdown(
                    label=_t("label_steps_md", "en"), elem_classes=["step-log"]
                )
                _i18n_outputs.append(steps_box)

                json_download = gr.File(label=_t("label_download_result", "en"))
                _i18n_outputs.append(json_download)

            def on_run(trap_id, model, agent_type, sandbox, thinking, effort, mutate):
                full_agent_type = (
                    agent_type if agent_type != "docker" else "langchain"
                )
                result = _run_evaluation(
                    trap_id, model, full_agent_type, sandbox,
                    thinking, effort, mutate,
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
                    trap_id_dropdown, model_input, agent_type, sandbox_input,
                    thinking_checkbox, effort_dropdown, mutate_checkbox,
                ],
                outputs=[compliance_box, hallucination_box, steps_box, json_download],
            )

        # ------------------------------------------------------------------
        # Tab 2: Report Viewer
        # ------------------------------------------------------------------
        with tab2:
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

        # ------------------------------------------------------------------
        # Tab 3: About
        # ------------------------------------------------------------------
        with tab3:
            about_md = gr.Markdown(_t("md_about", "en") + about_trailer)
            _i18n_outputs.append(about_md)

        # ------------------------------------------------------------------
        # Tab 4: Annotation
        # ------------------------------------------------------------------
        with tab4:
            _annotation_state = gr.State({
                "candidates": [],
                "annotations": [],
                "current_index": 0,
            })

            guide_accordion = gr.Accordion(
                _t("accordion_guide", "en"), open=False
            )
            _i18n_outputs.append(guide_accordion)
            with guide_accordion:
                guide_md = gr.Markdown(_t("md_annotation_guide", "en"))
                _i18n_outputs.append(guide_md)

            with gr.Row():
                with gr.Column(scale=1):
                    load_cand_md = gr.Markdown(_t("md_load_candidates", "en"))
                    _i18n_outputs.append(load_cand_md)

                    candidates_file = gr.File(
                        label=_t("label_upload_candidates", "en"),
                        file_types=[".json"],
                    )
                    _i18n_outputs.append(candidates_file)

                    output_path = gr.Textbox(
                        label=_t("label_output_path", "en"),
                        value="annotations.json",
                    )
                    _i18n_outputs.append(output_path)

                    load_status = gr.Textbox(
                        label=_t("label_status", "en"), interactive=False
                    )
                    _i18n_outputs.append(load_status)

                with gr.Column(scale=2):
                    nav_md = gr.Markdown(_t("md_navigation", "en"))
                    _i18n_outputs.append(nav_md)

                    with gr.Row():
                        prev_btn = gr.Button(_t("btn_previous", "en"), size="sm")
                        next_btn = gr.Button(
                            _t("btn_next", "en"), size="sm", variant="primary"
                        )
                        skip_btn = gr.Button(_t("btn_skip", "en"), size="sm")
                    _i18n_outputs.extend([prev_btn, next_btn, skip_btn])

                    progress_text = gr.Textbox(
                        label=_t("label_progress", "en"),
                        value="0 / 0",
                        interactive=False,
                    )
                    _i18n_outputs.append(progress_text)

            with gr.Row():
                with gr.Column(scale=3):
                    step_md = gr.Markdown(_t("md_step_content", "en"))
                    _i18n_outputs.append(step_md)

                    step_display = gr.Textbox(
                        label=_t("label_step_content", "en"), lines=8,
                        interactive=False,
                    )
                    _i18n_outputs.append(step_display)

                    metadata_display = gr.Textbox(
                        label=_t("label_metadata", "en"), lines=3,
                        interactive=False,
                    )
                    _i18n_outputs.append(metadata_display)

                with gr.Column(scale=2):
                    ev_md = gr.Markdown(_t("md_evidence", "en"))
                    _i18n_outputs.append(ev_md)

                    evidence_display = gr.Textbox(
                        label=_t("label_evidence", "en"), lines=6,
                        interactive=False,
                    )
                    _i18n_outputs.append(evidence_display)

                    explanation_display = gr.Textbox(
                        label=_t("label_explanation", "en"), lines=4,
                        interactive=False,
                    )
                    _i18n_outputs.append(explanation_display)

            ann_section_md = gr.Markdown(_t("md_annotation", "en"))
            _i18n_outputs.append(ann_section_md)

            with gr.Row():
                label_input = gr.Dropdown(
                    label=_t("label_gsar", "en"),
                    choices=GSAR_LABEL_OPTIONS,
                    value=None,
                    interactive=True,
                )
                _i18n_outputs.append(label_input)

                original_label_display = gr.Textbox(
                    label=_t("label_original", "en"), interactive=False
                )
                _i18n_outputs.append(original_label_display)

            with gr.Row():
                g_score_slider = gr.Slider(
                    label=_t("label_g_score", "en"), minimum=0.0, maximum=1.0,
                    value=0.5, step=0.01, info=_t("info_g_score", "en"),
                )
                u_score_slider = gr.Slider(
                    label=_t("label_u_score", "en"), minimum=0.0, maximum=1.0,
                    value=0.1, step=0.01, info=_t("info_u_score", "en"),
                )
            _i18n_outputs.extend([g_score_slider, u_score_slider])
            with gr.Row():
                c_score_slider = gr.Slider(
                    label=_t("label_c_score", "en"), minimum=0.0, maximum=1.0,
                    value=0.05, step=0.01, info=_t("info_c_score", "en"),
                )
                faithfulness_slider = gr.Slider(
                    label=_t("label_f_score", "en"), minimum=0.0, maximum=1.0,
                    value=0.9, step=0.01, info=_t("info_f_score", "en"),
                )
            _i18n_outputs.extend([c_score_slider, faithfulness_slider])

            submit_annotation_btn = gr.Button(
                _t("btn_submit", "en"), variant="primary"
            )
            _i18n_outputs.append(submit_annotation_btn)

            export_accordion = gr.Accordion(
                _t("accordion_export", "en"), open=False
            )
            _i18n_outputs.append(export_accordion)
            with export_accordion:
                export_btn = gr.Button(_t("btn_export", "en"))
                _i18n_outputs.append(export_btn)

                export_file = gr.File(label=_t("label_download_ann", "en"))
                _i18n_outputs.append(export_file)

                export_stats = gr.Textbox(
                    label=_t("label_ann_stats", "en"), lines=5,
                    interactive=False,
                )
                _i18n_outputs.append(export_stats)

            # -- annotation tab callbacks --

            def on_load_candidates(file, out_path):
                if file is None:
                    slider_no = _slider_updates(None, (0.5, 0.1, 0.05, 0.9))
                    return (
                        {"candidates": [], "annotations": [], "current_index": 0},
                        _t("no_file", "en"),
                        _t("label_step_content", "en"),
                        _t("label_metadata", "en"),
                        _t("label_evidence", "en"),
                        _t("label_explanation", "en"),
                        _t("label_original", "en"),
                        None,
                        slider_no[0], slider_no[1],
                        slider_no[2], slider_no[3],
                        "0 / 0",
                    )
                try:
                    with open(file.name, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    candidates = data.get("candidates", [])
                    meta = data.get("metadata", {})

                    annotations = []
                    if out_path and os.path.isfile(out_path):
                        with open(out_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        annotations = existing.get("annotations", [])

                    annotated_ids = set(
                        (a.get("trap_id"), a.get("step_index"))
                        for a in annotations
                    )

                    state = {
                        "candidates": candidates,
                        "annotations": annotations,
                        "current_index": 0,
                        "output_path": out_path,
                        "annotated_ids": annotated_ids,
                    }

                    status = (
                        f"Loaded {len(candidates)} candidates. "
                        f"Already annotated: {len(annotations)}. "
                        f"Source: {meta.get('source', 'unknown')}. "
                        f"Label dist: {meta.get('label_distribution', {})}"
                    )

                    if not candidates:
                        empty_sl = _slider_updates(None, (0.5, 0.1, 0.05, 0.9))
                        empty = (state, status,
                                 "", "", "", "", "",
                                 None,
                                 empty_sl[0], empty_sl[1],
                                 empty_sl[2], empty_sl[3],
                                 "0 / 0")
                        return empty

                    c = candidates[0]
                    step = c.get("step_content", "")[:3000]
                    meta_txt = (
                        f"Trap: {c.get('trap_id','?')} | "
                        f"Type: {c.get('trap_type','?')} | "
                        f"Step: {c.get('step_index','?')} "
                        f"({c.get('step_type','?')})"
                    )
                    ev = "\n".join(
                        e[:500] if isinstance(e, str) else str(e)[:500]
                        for e in (c.get("evidence", []) or [])
                    ) or _t("no_evidence", "en")
                    expl = (c.get("explanation", "") or "")[:800]
                    orig_label = c.get("original_gsar_label", "?")

                    preset = {
                        "Grounded": (0.90, 0.05, 0.05, 0.95),
                        "Ungrounded": (0.10, 0.85, 0.10, 0.15),
                        "Contradicted": (0.05, 0.10, 0.90, 0.10),
                        "Complementary": (0.80, 0.10, 0.05, 0.90),
                    }
                    ps = preset.get(orig_label, (0.5, 0.1, 0.05, 0.9))
                    display_label = (
                        orig_label if orig_label in GSAR_LABEL_OPTIONS else None
                    )
                    load_sl = _slider_updates(display_label, ps)

                    return (
                        state, status,
                        step, meta_txt,
                        ev, expl,
                        orig_label,
                        display_label,
                        load_sl[0], load_sl[1],
                        load_sl[2], load_sl[3],
                        f"1 / {len(candidates)}",
                    )
                except Exception as e:
                    err_sl = _slider_updates(None, (0.5, 0.1, 0.05, 0.9))
                    return (
                        {"candidates": [], "annotations": [], "current_index": 0},
                        f"{_t('error_prefix', 'en')}{e}",
                        "", "", "", "", "",
                        None,
                        err_sl[0], err_sl[1],
                        err_sl[2], err_sl[3],
                        "0 / 0",
                    )

            def on_navigate(state, direction, output_path):
                candidates = state.get("candidates", [])
                if not candidates:
                    return _empty_step_results(state)

                total = len(candidates)
                current = state.get("current_index", 0)

                if direction == "next":
                    current = min(current + 1, total - 1)
                elif direction == "prev":
                    current = max(current - 1, 0)
                elif direction == "skip":
                    current = min(current + 1, total - 1)

                state["current_index"] = current
                return _render_candidate(state)

            def on_submit_annotation(
                state, label, g_score, u_score, c_score, faithfulness
            ):
                candidates = state.get("candidates", [])
                current = state.get("current_index", 0)
                if not candidates or label is None:
                    return _render_candidate(state)

                c = candidates[current]
                key = (c.get("trap_id"), c.get("step_index"))

                annotations = state.get("annotations", [])
                annotated_ids = state.get("annotated_ids", set())

                if key in annotated_ids:
                    for a in annotations:
                        if (a.get("trap_id") == key[0]
                                and a.get("step_index") == key[1]):
                            a["gsar_label"] = label
                            a["g_score"] = g_score
                            a["u_score"] = u_score
                            a["c_score"] = c_score
                            a["faithfulness_score"] = faithfulness
                            break
                else:
                    annotations.append({
                        "trap_id": key[0],
                        "step_index": key[1],
                        "gsar_label": label,
                        "g_score": g_score,
                        "u_score": u_score,
                        "c_score": c_score,
                        "faithfulness_score": faithfulness,
                    })
                    annotated_ids.add(key)

                state["annotations"] = annotations
                state["annotated_ids"] = annotated_ids

                auto_path = state.get("output_path", "annotations.json")
                _save_annotations_web(auto_path, annotations)

                current = min(current + 1, len(candidates) - 1)
                state["current_index"] = current
                return _render_candidate(state)

            def on_export(state):
                out_path = state.get("output_path", "annotations.json")
                annotations = state.get("annotations", [])
                _save_annotations_web(out_path, annotations)

                candidates = state.get("candidates", [])
                total = len(candidates)
                annotated = len(annotations)
                label_counts = {}
                for a in annotations:
                    lab = a.get("gsar_label", "?")
                    label_counts[lab] = label_counts.get(lab, 0) + 1

                stats = (
                    f"Candidates: {total}\n"
                    f"Annotated: {annotated}\n"
                    f"Remaining: {total - annotated}\n"
                    f"Labels: {label_counts}"
                )
                return out_path, stats

            def on_label_change(label):
                preset = {
                    "Grounded": (0.90, 0.05, 0.05, 0.95),
                    "Ungrounded": (0.10, 0.85, 0.10, 0.15),
                    "Contradicted": (0.05, 0.10, 0.90, 0.10),
                    "Complementary": (0.80, 0.10, 0.05, 0.90),
                }
                ps = preset.get(label, (0.5, 0.1, 0.05, 0.9))
                return _slider_updates(label, ps)

            candidates_file.change(
                fn=on_load_candidates,
                inputs=[candidates_file, output_path],
                outputs=[
                    _annotation_state, load_status,
                    step_display, metadata_display,
                    evidence_display, explanation_display,
                    original_label_display,
                    label_input, g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                    progress_text,
                ],
            )

            prev_btn.click(
                fn=lambda s, p: on_navigate(s, "prev", p),
                inputs=[_annotation_state, output_path],
                outputs=[
                    _annotation_state, load_status,
                    step_display, metadata_display,
                    evidence_display, explanation_display,
                    original_label_display,
                    label_input, g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                    progress_text,
                ],
            )
            next_btn.click(
                fn=lambda s, p: on_navigate(s, "next", p),
                inputs=[_annotation_state, output_path],
                outputs=[
                    _annotation_state, load_status,
                    step_display, metadata_display,
                    evidence_display, explanation_display,
                    original_label_display,
                    label_input, g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                    progress_text,
                ],
            )
            skip_btn.click(
                fn=lambda s, p: on_navigate(s, "skip", p),
                inputs=[_annotation_state, output_path],
                outputs=[
                    _annotation_state, load_status,
                    step_display, metadata_display,
                    evidence_display, explanation_display,
                    original_label_display,
                    label_input, g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                    progress_text,
                ],
            )

            submit_annotation_btn.click(
                fn=on_submit_annotation,
                inputs=[
                    _annotation_state, label_input, g_score_slider,
                    u_score_slider, c_score_slider, faithfulness_slider,
                ],
                outputs=[
                    _annotation_state, load_status,
                    step_display, metadata_display,
                    evidence_display, explanation_display,
                    original_label_display,
                    label_input, g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                    progress_text,
                ],
            )

            label_input.change(
                fn=on_label_change,
                inputs=[label_input],
                outputs=[
                    g_score_slider, u_score_slider,
                    c_score_slider, faithfulness_slider,
                ],
            )

            export_btn.click(
                fn=on_export,
                inputs=[_annotation_state],
                outputs=[export_file, export_stats],
            )

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
                gr.update(label=_t("label_trap_id", lang),
                          info=_t("info_select_trap", lang)),
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
                gr.update(label=_t("label_g_score", lang),
                           info=_t("info_g_score", lang)),
                gr.update(label=_t("label_u_score", lang),
                           info=_t("info_u_score", lang)),
                gr.update(label=_t("label_c_score", lang),
                           info=_t("info_c_score", lang)),
                gr.update(label=_t("label_f_score", lang),
                           info=_t("info_f_score", lang)),
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
    demo = create_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme="soft",
        css="""
        :root {
            --font-sans: "Inter", -apple-system, BlinkMacSystemFont,
                "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB",
                "Noto Sans SC", "Segoe UI", Roboto, "Helvetica Neue",
                Arial, sans-serif;
            --font-mono: "JetBrains Mono", "Fira Code", "SF Mono",
                "Cascadia Code", "Source Code Pro", Menlo, monospace;
        }
        * {
            font-family: var(--font-sans) !important;
        }
        code, pre, .prose code, .prose pre {
            font-family: var(--font-mono) !important;
        }
        .step-log { max-height: 400px; overflow-y: auto; }
        .result-box { padding: 10px; border-radius: 5px; }
        #lang-selector {
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 0.5rem;
        }
        #lang-selector label {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            cursor: pointer;
        }
        """,
    )
