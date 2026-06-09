"""Tab 4: Annotation — interactive GSAR labeling with slider presets and auto-save."""

import json
import os

import gradio as gr

from agent_trust_lab.web._shared import (
    GSAR_LABEL_OPTIONS,
    _LABEL_SCORE_PRESETS,
    _empty_step_results,
    _render_candidate,
    _save_annotations_web,
    _slider_updates,
)
from agent_trust_lab.web.i18n import t as _t


def build_annotation_tab(_i18n_outputs: list, **kwargs) -> None:
    """Build the Annotation tab inside the current Gradio context.

    Args:
        _i18n_outputs: Shared list — append every Gradio component here
            in creation order so the language-switch callback can update them.
    """
    _annotation_state = gr.State(
        {
            "candidates": [],
            "annotations": [],
            "current_index": 0,
        }
    )

    guide_accordion = gr.Accordion(_t("accordion_guide", "en"), open=False)
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
                label=_t("label_step_content", "en"),
                lines=8,
                interactive=False,
            )
            _i18n_outputs.append(step_display)

            metadata_display = gr.Textbox(
                label=_t("label_metadata", "en"),
                lines=3,
                interactive=False,
            )
            _i18n_outputs.append(metadata_display)

        with gr.Column(scale=2):
            ev_md = gr.Markdown(_t("md_evidence", "en"))
            _i18n_outputs.append(ev_md)

            evidence_display = gr.Textbox(
                label=_t("label_evidence", "en"),
                lines=6,
                interactive=False,
            )
            _i18n_outputs.append(evidence_display)

            explanation_display = gr.Textbox(
                label=_t("label_explanation", "en"),
                lines=4,
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
            label=_t("label_g_score", "en"),
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            info=_t("info_g_score", "en"),
        )
        u_score_slider = gr.Slider(
            label=_t("label_u_score", "en"),
            minimum=0.0,
            maximum=1.0,
            value=0.1,
            step=0.01,
            info=_t("info_u_score", "en"),
        )
    _i18n_outputs.extend([g_score_slider, u_score_slider])
    with gr.Row():
        c_score_slider = gr.Slider(
            label=_t("label_c_score", "en"),
            minimum=0.0,
            maximum=1.0,
            value=0.05,
            step=0.01,
            info=_t("info_c_score", "en"),
        )
        faithfulness_slider = gr.Slider(
            label=_t("label_f_score", "en"),
            minimum=0.0,
            maximum=1.0,
            value=0.9,
            step=0.01,
            info=_t("info_f_score", "en"),
        )
    _i18n_outputs.extend([c_score_slider, faithfulness_slider])

    submit_annotation_btn = gr.Button(
        _t("btn_submit", "en"), variant="primary"
    )
    _i18n_outputs.append(submit_annotation_btn)

    export_accordion = gr.Accordion(_t("accordion_export", "en"), open=False)
    _i18n_outputs.append(export_accordion)
    with export_accordion:
        export_btn = gr.Button(_t("btn_export", "en"))
        _i18n_outputs.append(export_btn)

        export_file = gr.File(label=_t("label_download_ann", "en"))
        _i18n_outputs.append(export_file)

        export_stats = gr.Textbox(
            label=_t("label_ann_stats", "en"),
            lines=5,
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
                slider_no[0],
                slider_no[1],
                slider_no[2],
                slider_no[3],
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
                (a.get("trap_id"), a.get("step_index")) for a in annotations
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
                empty = (
                    state,
                    status,
                    "",
                    "",
                    "",
                    "",
                    "",
                    None,
                    empty_sl[0],
                    empty_sl[1],
                    empty_sl[2],
                    empty_sl[3],
                    "0 / 0",
                )
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
            display_label = orig_label if orig_label in GSAR_LABEL_OPTIONS else None
            load_sl = _slider_updates(display_label, ps)

            return (
                state,
                status,
                step,
                meta_txt,
                ev,
                expl,
                orig_label,
                display_label,
                load_sl[0],
                load_sl[1],
                load_sl[2],
                load_sl[3],
                f"1 / {len(candidates)}",
            )
        except Exception as e:
            err_sl = _slider_updates(None, (0.5, 0.1, 0.05, 0.9))
            return (
                {"candidates": [], "annotations": [], "current_index": 0},
                f"{_t('error_prefix', 'en')}{e}",
                "",
                "",
                "",
                "",
                "",
                None,
                err_sl[0],
                err_sl[1],
                err_sl[2],
                err_sl[3],
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

    def on_submit_annotation(state, label, g_score, u_score, c_score, faithfulness):
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
                if a.get("trap_id") == key[0] and a.get("step_index") == key[1]:
                    a["gsar_label"] = label
                    a["g_score"] = g_score
                    a["u_score"] = u_score
                    a["c_score"] = c_score
                    a["faithfulness_score"] = faithfulness
                    break
        else:
            annotations.append(
                {
                    "trap_id": key[0],
                    "step_index": key[1],
                    "gsar_label": label,
                    "g_score": g_score,
                    "u_score": u_score,
                    "c_score": c_score,
                    "faithfulness_score": faithfulness,
                }
            )
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
            _annotation_state,
            load_status,
            step_display,
            metadata_display,
            evidence_display,
            explanation_display,
            original_label_display,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
            progress_text,
        ],
    )

    prev_btn.click(
        fn=lambda s, p: on_navigate(s, "prev", p),
        inputs=[_annotation_state, output_path],
        outputs=[
            _annotation_state,
            load_status,
            step_display,
            metadata_display,
            evidence_display,
            explanation_display,
            original_label_display,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
            progress_text,
        ],
    )
    next_btn.click(
        fn=lambda s, p: on_navigate(s, "next", p),
        inputs=[_annotation_state, output_path],
        outputs=[
            _annotation_state,
            load_status,
            step_display,
            metadata_display,
            evidence_display,
            explanation_display,
            original_label_display,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
            progress_text,
        ],
    )
    skip_btn.click(
        fn=lambda s, p: on_navigate(s, "skip", p),
        inputs=[_annotation_state, output_path],
        outputs=[
            _annotation_state,
            load_status,
            step_display,
            metadata_display,
            evidence_display,
            explanation_display,
            original_label_display,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
            progress_text,
        ],
    )

    submit_annotation_btn.click(
        fn=on_submit_annotation,
        inputs=[
            _annotation_state,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
        ],
        outputs=[
            _annotation_state,
            load_status,
            step_display,
            metadata_display,
            evidence_display,
            explanation_display,
            original_label_display,
            label_input,
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
            progress_text,
        ],
    )

    label_input.change(
        fn=on_label_change,
        inputs=[label_input],
        outputs=[
            g_score_slider,
            u_score_slider,
            c_score_slider,
            faithfulness_slider,
        ],
    )

    export_btn.click(
        fn=on_export,
        inputs=[_annotation_state],
        outputs=[export_file, export_stats],
    )
