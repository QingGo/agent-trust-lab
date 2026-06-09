"""Batch evaluation support for comparing multiple agent configurations.

Provides YAML config parsing and a batch runner that executes multiple
evaluation runs, merges results, and generates a comparison report.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agent_trust_lab.config import DEFAULT_MODEL, EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.orchestrator import EvaluationResult, Orchestrator

logger = get_logger("batch")


@dataclass
class EvaluationSpec:
    """A single evaluation specification from batch YAML config."""

    label: str
    model: str = DEFAULT_MODEL
    agent_type: str = "langchain"
    thinking_enabled: bool = False
    reasoning_effort: str = ""
    base_url: str = ""
    api_key: str = ""
    skip_hallukg: bool = False
    cache_enabled: bool = True
    judge_model: str = ""
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
    concurrent: bool = False
    output_dir: str = "./results/"
    report_format: str = "html"
    report_lang: str = "en"
    report_open: bool = False
    report_url: str = ""
    calibration_profile: Optional[str] = None
    timeout: int = 120
    runs: int = 1


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
                model=str(ev.get("model", DEFAULT_MODEL)),
                agent_type=str(ev.get("agent_type", "langchain")),
                thinking_enabled=bool(ev.get("thinking_enabled", False)),
                reasoning_effort=str(ev.get("reasoning_effort", "")),
                base_url=str(ev.get("base_url", "")),
                api_key=str(ev.get("api_key", "")),
                skip_hallukg=bool(ev.get("skip_hallukg", False)),
                cache_enabled=bool(ev.get("cache_enabled", True)),
                judge_model=str(ev.get("judge_model", "")),
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
        concurrent=bool(common.get("concurrent", False)),
        output_dir=str(common.get("output_dir", "./results/")),
        report_format=str(report_cfg.get("format", "html")).lower(),
        report_lang=str(report_cfg.get("lang", "en")).lower(),
        report_open=bool(report_cfg.get("open", False)),
        report_url=str(report_cfg.get("url", "")),
        calibration_profile=common.get("calibration_profile"),
        timeout=int(common.get("timeout", 120)),
        runs=int(common.get("runs", 1)),
    )


def _run_single_eval(
    spec: EvaluationSpec, batch_config: "BatchConfig"
) -> Tuple[str, List[EvaluationResult], str]:
    """Run a single evaluation spec and return (safe_label, results, json_path).

    Skips evaluation if the output JSON already exists with valid results.
    """
    safe_label = spec.label.replace(" ", "_").replace("/", "_")
    json_path = os.path.join(batch_config.output_dir, f"{safe_label}.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            existing = data.get("results", data)
            if isinstance(existing, list) and len(existing) > 0:
                has_runs = any(r.get("runs_count", 0) >= batch_config.runs for r in existing[:3])
                if has_runs or batch_config.runs <= 1:
                    logger.info(
                        "  -> Skipping %s (already completed: %d results)",
                        safe_label, len(existing),
                    )
                    return safe_label, [], json_path
        except (json.JSONDecodeError, OSError) as e:
            pass

    config_kwargs: Dict[str, Any] = dict(
        model=spec.model,
        agent_type=spec.agent_type,
        thinking_enabled=spec.thinking_enabled,
        reasoning_effort=spec.reasoning_effort,
        base_url=spec.base_url,
        api_key=spec.api_key,
        skip_hallukg=spec.skip_hallukg,
        cache_enabled=spec.cache_enabled,
        sandbox=batch_config.sandbox,
        sandbox_image=batch_config.sandbox_image,
        sandbox_network=batch_config.sandbox_network,
        docker_host=batch_config.docker_host,
        parallel=batch_config.parallel,
        trap_library_path=batch_config.trap_library_path,
        calibration_profile=batch_config.calibration_profile,
        timeout=batch_config.timeout,
        runs=batch_config.runs,
    )
    if spec.judge_model:
        config_kwargs["judge_model"] = spec.judge_model
    config = EvaluationConfig(**config_kwargs)

    orch = Orchestrator(config)
    results = list(orch.run_traps(**spec.traps))
    orch.export_results(results, json_path)
    logger.info("  -> %d results saved to %s", len(results), json_path)

    return safe_label, results, json_path


def run_batch(batch_config: BatchConfig) -> Dict[str, Any]:
    """Execute all evaluations in a batch configuration.

    When concurrent=True, evaluations run in parallel via ThreadPoolExecutor.
    Each evaluation creates its own Orchestrator instance.
    Trap-level parallelism within each evaluation is controlled by batch_config.parallel.

    Args:
        batch_config: Parsed BatchConfig from parse_batch_yaml().

    Returns:
        The merged multi-model results dict usable by ReportGenerator.
    """
    os.makedirs(batch_config.output_dir, exist_ok=True)

    json_paths: List[str] = []

    if batch_config.concurrent and len(batch_config.evaluations) > 1:
        logger.info(
            "Running %d evaluations concurrently (max_workers=%d)",
            len(batch_config.evaluations),
            len(batch_config.evaluations),
        )
        with ThreadPoolExecutor(max_workers=len(batch_config.evaluations)) as executor:
            futures = {
                executor.submit(_run_single_eval, spec, batch_config): i
                for i, spec in enumerate(batch_config.evaluations)
            }
            for future in futures:
                try:
                    safe_label, results, json_path = future.result()
                    logger.info(
                        "Completed: %s (model=%s) — %d traps",
                        safe_label,
                        batch_config.evaluations[futures[future]].model,
                        len(results),
                    )
                    json_paths.append(json_path)
                except Exception as e:
                    idx = futures[future]
                    logger.error(
                        "Evaluation '%s' failed: %s", batch_config.evaluations[idx].label, e
                    )
    else:
        for spec in batch_config.evaluations:
            logger.info("Running: %s (model=%s, agent=%s)", spec.label, spec.model, spec.agent_type)
            try:
                safe_label, results, json_path = _run_single_eval(spec, batch_config)
                json_paths.append(json_path)
            except Exception as e:
                logger.error(
                    "Evaluation '%s' failed: %s. "
                    "Check model name, API key, and trap configuration.",
                    spec.label, e,
                )
                continue

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
        if batch_config.report_lang == "both":
            generator.generate_both(
                merged, batch_config.output_dir, "comparison", cal_data,
                report_url=batch_config.report_url,
            )
            report_path = os.path.join(batch_config.output_dir, "comparison.html")
        else:
            report_path = os.path.join(batch_config.output_dir, "comparison.html")
            generator.generate(
                merged,
                output_path=report_path,
                calibration=cal_data,
                lang=batch_config.report_lang,
                report_url=batch_config.report_url,
            )
    logger.info("Comparison report saved to %s", report_path)

    if batch_config.report_open and batch_config.report_format == "html":
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(report_path)}")

    return merged
