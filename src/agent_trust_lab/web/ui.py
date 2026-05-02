import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr


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


def _format_trajectory_steps(result: Dict[str, Any]) -> str:
    lines = []
    steps = result.get("trajectory_steps", [])
    if not steps:
        return "No trajectory data available"

    for i, step in enumerate(steps):
        step_type = step.get("type", "unknown")
        content = step.get("content", "")
        lines.append(f"### Step {i}: {step_type}")
        lines.append(f"```\n{content[:500]}\n```")
        lines.append("---")
    return "\n".join(lines)


def create_ui() -> gr.Blocks:
    trap_choices = _build_trap_choices()
    all_trap_ids = []
    for ids in trap_choices.values():
        all_trap_ids.extend(ids)
    all_trap_ids = sorted(set(all_trap_ids))

    categories = sorted(trap_choices.keys())

    with gr.Blocks(
        title="Agent Trust Lab",
        theme="soft",
        css="""
        .step-log { max-height: 400px; overflow-y: auto; }
        .result-box { padding: 10px; border-radius: 5px; }
    """,
    ) as demo:
        gr.Markdown(
            "# Agent Trust Lab\n"
            "LLM Agent Reliability & Hallucination Evaluation Platform"
        )

        with gr.Tabs():
            with gr.TabItem("Run Evaluation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Trap Selection")
                        trap_category = gr.Dropdown(
                            label="Category",
                            choices=sorted(categories),
                            value=categories[0] if categories else None,
                        )
                        trap_id_dropdown = gr.Dropdown(
                            label="Trap ID",
                            choices=all_trap_ids,
                            info="Select a trap to evaluate",
                        )
                        with gr.Accordion("Trap Details", open=False):
                            trap_info = gr.JSON(label="Trap Info")

                    with gr.Column(scale=1):
                        gr.Markdown("### Agent Configuration")
                        model_input = gr.Textbox(
                            label="Model", value="deepseek-v4-flash"
                        )
                        agent_type = gr.Dropdown(
                            label="Harness",
                            choices=["langchain", "codex", "openai", "opencode",
                                     "claude-code", "gemini-cli"],
                            value="langchain",
                        )
                        sandbox_input = gr.Dropdown(
                            label="Sandbox",
                            choices=["docker", "dry-run"],
                            value="docker",
                        )
                        thinking_checkbox = gr.Checkbox(
                            label="Thinking Mode", value=False
                        )
                        effort_dropdown = gr.Dropdown(
                            label="Reasoning Effort",
                            choices=["", "high", "max"],
                            value="",
                            visible=False,
                        )
                        mutate_checkbox = gr.Checkbox(
                            label="Apply Mutation", value=False
                        )

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
                    info = _get_trap_info(trap_id)
                    return info

                trap_id_dropdown.change(
                    fn=on_trap_select,
                    inputs=[trap_id_dropdown],
                    outputs=[trap_info],
                )

                run_button = gr.Button("Run Evaluation", variant="primary", size="lg")

                with gr.Accordion("Results", open=True):
                    compliance_box = gr.Textbox(
                        label="Compliance Status", lines=3
                    )
                    hallucination_box = gr.JSON(label="Hallucination Scores")
                    steps_box = gr.Markdown(
                        label="Trajectory Steps", elem_classes=["step-log"]
                    )
                    json_download = gr.File(label="Download Results JSON")

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

            with gr.TabItem("Report Viewer"):
                gr.Markdown("### Load and View Results")
                report_json = gr.File(
                    label="Upload Results JSON",
                    file_types=[".json"],
                )
                report_content = gr.JSON(label="Report Content")

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

            with gr.TabItem("About"):
                gr.Markdown(
                    "## Agent Trust Lab\n\n"
                    "A modular platform for evaluating LLM agent reliability through:\n\n"
                    "- **Trap injection** — injecting hallucination-inducing prompts\n"
                    "- **Harness execution** — running agents with real LLM APIs\n"
                    "- **Trajectory capture** — recording step-by-step agent behavior\n"
                    "- **Multi-hop hallucination detection** — ONNX + NetworkX grounding\n"
                    "- **Compliance audit** — 10 PAE rules across 6+4 dimensions\n"
                    "- **Calibrated reporting** — Platt scaling + Cohen's kappa\n\n"
                    f"**Available traps:** {len(all_trap_ids)} "
                    f"({len(trap_choices)} categories)\n\n"
                    "**8 registered harnesses:** langchain, codex, openai, opencode, "
                    "claude-code, gemini-cli, docker, dry-run\n\n"
                    "**2 report formats:** HTML, Markdown (en/zh)"
                )

    return demo


def launch_ui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    demo = create_ui()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
