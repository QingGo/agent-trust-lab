"""Batch evaluation support for comparing multiple agent configurations.

Provides YAML config parsing and a batch runner that executes multiple
evaluation runs, merges results, and generates a comparison report.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.orchestrator import Orchestrator

logger = get_logger("batch")


@dataclass
class EvaluationSpec:
    """A single evaluation specification from batch YAML config."""

    label: str
    model: str = "deepseek-v4-flash"
    agent_type: str = "langchain"
    thinking_enabled: bool = False
    reasoning_effort: str = ""
    base_url: str = ""
    api_key: str = ""
    skip_hallukg: bool = False
    traps: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Parsed batch evaluation configuration."""

    evaluations: List[EvaluationSpec]
    trap_library_path: str = ""
    sandbox: str = "docker"
    sandbox_image: str = ""
    sandbox_network: bool = False
    docker_host: str = ""
    parallel: int = 1
    output_dir: str = "./results/"
    report_format: str = "html"
    report_lang: str = "en"
    report_open: bool = False
    calibration_profile: Optional[str] = None
    timeout: int = 120


def parse_batch_yaml(yaml_path: str) -> BatchConfig:
    """Parse a batch YAML configuration file.

    Args:
        yaml_path: Path to the batch YAML file.

    Returns:
        A BatchConfig with evaluation specs and common settings.

    Raises:
        ValueError: If the YAML is invalid or missing required fields.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Batch YAML must be a mapping at the top level.")

    if "evaluations" not in raw:
        raise ValueError("Batch YAML must contain an 'evaluations' key.")
    if not isinstance(raw["evaluations"], list) or len(raw["evaluations"]) == 0:
        raise ValueError("'evaluations' must be a non-empty list.")

    common = raw.get("common", {}) or {}
    report_cfg = raw.get("report", {}) or {}

    eval_specs: List[EvaluationSpec] = []
    for i, ev in enumerate(raw["evaluations"]):
        if not isinstance(ev, dict):
            raise ValueError(f"Evaluations[{i}] must be a mapping.")
        label = str(ev.get("label", ""))
        if not label:
            raise ValueError(f"Evaluations[{i}] is missing required 'label' field.")

        traps = ev.get("traps", {}) or {}
        if not isinstance(traps, dict):
            raise ValueError(f"Evaluations[{i}] 'traps' must be a mapping.")

        eval_specs.append(
            EvaluationSpec(
                label=label,
                model=str(ev.get("model", "deepseek-v4-flash")),
                agent_type=str(ev.get("agent_type", "langchain")),
                thinking_enabled=bool(ev.get("thinking_enabled", False)),
                reasoning_effort=str(ev.get("reasoning_effort", "")),
                base_url=str(ev.get("base_url", "")),
                api_key=str(ev.get("api_key", "")),
                skip_hallukg=bool(ev.get("skip_hallukg", False)),
                traps=traps,
            )
        )

    return BatchConfig(
        evaluations=eval_specs,
        trap_library_path=str(common.get("trap_library_path", "")),
        sandbox=str(common.get("sandbox", "docker")),
        sandbox_image=str(common.get("sandbox_image", "")),
        sandbox_network=bool(common.get("sandbox_network", False)),
        docker_host=str(common.get("docker_host", "")),
        parallel=int(common.get("parallel", 1)),
        output_dir=str(common.get("output_dir", "./results/")),
        report_format=str(report_cfg.get("format", "html")).lower(),
        report_lang=str(report_cfg.get("lang", "en")).lower(),
        report_open=bool(report_cfg.get("open", False)),
        calibration_profile=common.get("calibration_profile"),
        timeout=int(common.get("timeout", 120)),
    )


def run_batch(batch_config: BatchConfig) -> Dict[str, Any]:
    """Execute all evaluations in a batch configuration serially.

    Each evaluation specification is run in sequence. Results are saved
    as individual JSON files, then merged into a multi-model comparison
    report.

    Args:
        batch_config: Parsed BatchConfig from parse_batch_yaml().

    Returns:
        The merged multi-model results dict usable by ReportGenerator.
    """
    os.makedirs(batch_config.output_dir, exist_ok=True)

    json_paths: List[str] = []
    for spec in batch_config.evaluations:
        logger.info("Running: %s (model=%s, agent=%s)", spec.label, spec.model, spec.agent_type)

        config = EvaluationConfig(
            model=spec.model,
            agent_type=spec.agent_type,
            thinking_enabled=spec.thinking_enabled,
            reasoning_effort=spec.reasoning_effort,
            base_url=spec.base_url,
            api_key=spec.api_key,
            skip_hallukg=spec.skip_hallukg,
            sandbox=batch_config.sandbox,
            sandbox_image=batch_config.sandbox_image,
            sandbox_network=batch_config.sandbox_network,
            docker_host=batch_config.docker_host,
            parallel=batch_config.parallel,
            trap_library_path=batch_config.trap_library_path,
            calibration_profile=batch_config.calibration_profile,
            timeout=batch_config.timeout,
        )

        orch = Orchestrator(config)
        results = orch.run_traps(**spec.traps)

        safe_label = spec.label.replace(" ", "_").replace("/", "_")
        json_path = os.path.join(batch_config.output_dir, f"{safe_label}.json")
        orch.export_results(results, json_path)
        json_paths.append(json_path)
        logger.info("  -> %d results saved to %s", len(results), json_path)

    if len(json_paths) < 2:
        logger.warning("Only 1 evaluation run; comparison report requires >= 2.")

    from agent_trust_lab.report import ReportGenerator

    merged = ReportGenerator.merge_results(json_paths)
    merged_path = os.path.join(batch_config.output_dir, "comparison.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged comparison data saved to %s", merged_path)

    generator = ReportGenerator()
    cal_data = None
    if batch_config.calibration_profile:
        from agent_trust_lab.calibration.profile import load_profile

        profile = load_profile(batch_config.calibration_profile)
        if profile:
            cal_data = profile.to_dict()

    if batch_config.report_format in ("markdown", "md"):
        report_path = os.path.join(batch_config.output_dir, "comparison.md")
        generator.generate_markdown(
            merged,
            output_path=report_path,
            calibration=cal_data,
            lang=batch_config.report_lang,
        )
    else:
        report_path = os.path.join(batch_config.output_dir, "comparison.html")
        generator.generate(
            merged,
            output_path=report_path,
            calibration=cal_data,
            lang=batch_config.report_lang,
        )
    logger.info("Comparison report saved to %s", report_path)

    if batch_config.report_open and batch_config.report_format == "html":
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(report_path)}")

    return merged
