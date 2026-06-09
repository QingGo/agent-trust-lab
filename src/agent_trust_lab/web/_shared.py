"""Shared constants, helpers, CSS, and trap utilities for the web UI."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

GSAR_LABEL_OPTIONS = ["Grounded", "Ungrounded", "Contradicted", "Complementary"]

_LABEL_SCORE_PRESETS = {
    "Grounded": (0.90, 0.05, 0.05, 0.95),
    "Ungrounded": (0.10, 0.85, 0.10, 0.15),
    "Contradicted": (0.05, 0.10, 0.90, 0.10),
    "Complementary": (0.80, 0.10, 0.05, 0.90),
}

_SLIDER_CONSTRAINTS: dict = {
    "Grounded": (
        dict(minimum=0.50, maximum=1.0),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.50, maximum=1.0),
    ),
    "Ungrounded": (
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.50, maximum=1.0),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),
    ),
    "Contradicted": (
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.50, maximum=1.0),
        dict(minimum=0.0, maximum=0.49),
    ),
    "Complementary": (
        dict(minimum=0.50, maximum=1.0),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.0, maximum=0.49),
        dict(minimum=0.50, maximum=1.0),
    ),
    "none": (
        dict(minimum=0.0, maximum=1.0),
        dict(minimum=0.0, maximum=1.0),
        dict(minimum=0.0, maximum=1.0),
        dict(minimum=0.0, maximum=1.0),
    ),
}

UI_CSS = """
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
"""


def _slider_updates(label: str | None, values: tuple) -> list:
    """Build gr.update() list for 4 sliders with label-based constraints."""
    if label and label in _SLIDER_CONSTRAINTS:
        constraints = _SLIDER_CONSTRAINTS[label]
    else:
        constraints = _SLIDER_CONSTRAINTS["none"]
    return [
        _gr_update(value=values[0], **constraints[0]),
        _gr_update(value=values[1], **constraints[1]),
        _gr_update(value=values[2], **constraints[2]),
        _gr_update(value=values[3], **constraints[3]),
    ]


def _gr_update(**kwargs):
    """Lazy import of gradio update to avoid import-time dependency."""
    import gradio as gr

    return gr.update(**kwargs)


def _save_annotations_web(output_path: str, annotations: list) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = {"benchmark": "gsar-calibration-v1", "version": "1.0", "annotations": annotations}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def _render_candidate(state: dict, lang: str = "en") -> tuple:
    from agent_trust_lab.web.i18n import t as _t

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
                    a.get("g_score", 0.5),
                    a.get("u_score", 0.1),
                    a.get("c_score", 0.05),
                    a.get("faithfulness_score", 0.9),
                )
                preset = _LABEL_SCORE_PRESETS.get(label, default_scores)
                slider_vals = _slider_updates(label, preset)
                return (
                    state,
                    status_text,
                    step,
                    meta_txt,
                    ev,
                    expl,
                    orig_label,
                    label,
                    slider_vals[0],
                    slider_vals[1],
                    slider_vals[2],
                    slider_vals[3],
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
        state,
        status_text,
        step,
        meta_txt,
        ev,
        expl,
        orig_label,
        displayed_label,
        slider_vals[0],
        slider_vals[1],
        slider_vals[2],
        slider_vals[3],
        status_text,
    )


def _empty_step_results(state: dict, message: str = "No data") -> tuple:
    total = len(state.get("candidates", []))
    progress = f"{total} / {total}" if total else "0 / 0"
    default = (0.5, 0.1, 0.05, 0.9)
    slider_vals = _slider_updates(None, default)
    return (
        state,
        message,
        "",
        "",
        "",
        "",
        "",
        None,
        slider_vals[0],
        slider_vals[1],
        slider_vals[2],
        slider_vals[3],
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
