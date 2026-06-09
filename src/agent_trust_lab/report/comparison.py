"""Multi-model comparison report generator for agent-trust-lab."""

import json
from typing import Any, Dict, List


class ComparisonReportGenerator:
    """Generates multi-model comparison data from single-model result files."""

    @staticmethod
    def merge_results(json_paths: List[str]) -> Dict[str, Any]:
        """Merge multiple single-model results.json files into multi-model format."""
        configs: List[Dict[str, Any]] = []
        all_trap_ids: List[str] = []
        model_results: Dict[str, List[Dict[str, Any]]] = {}

        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            model_label = cfg.get("model", "unknown")
            if cfg.get("thinking_enabled"):
                model_label += f" (think {cfg.get('reasoning_effort', 'high')})"
            else:
                model_label += " (no-think)"
            config_entry = {
                "model": cfg.get("model", ""),
                "thinking_enabled": cfg.get("thinking_enabled", False),
                "reasoning_effort": cfg.get("reasoning_effort", ""),
                "config_label": model_label,
            }
            # Deduplicate: avoid adding the same config twice when the same
            # model configuration is used against different trap categories.
            if model_label not in {c["config_label"] for c in configs}:
                configs.append(config_entry)
            for r in data.get("results", []):
                tid = r.get("trap_id", "")
                if tid not in model_results:
                    model_results[tid] = []
                    all_trap_ids.append(tid)
                model_results[tid].append({"label": model_label, "data": r})

        traps_meta: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        for tid in all_trap_ids:
            entries = model_results.get(tid, [])
            base_meta = {}
            for entry in entries:
                r = entry["data"]
                if not base_meta:
                    meta = r.get("metadata", {})
                    base_meta = {
                        "base_task": meta.get("base_task", ""),
                        "trap_injection": meta.get("trap_injection", ""),
                        "knowledge_source": meta.get("knowledge_source", ""),
                        "severity": meta.get("severity", ""),
                        "difficulty": meta.get("difficulty", ""),
                    }
                    traps_meta.append(
                        {
                            "trap_id": tid,
                            "trap_type": r.get("trap_type", ""),
                            "category": r.get("category", ""),
                            "severity": meta.get("severity", ""),
                            "base_task": meta.get("base_task", ""),
                            "trap_injection": meta.get("trap_injection", ""),
                            "knowledge_source": meta.get("knowledge_source", ""),
                        }
                    )
            scores = {}
            for entry in entries:
                scores[entry["label"]] = entry["data"]
            results.append(
                {
                    "trap_id": tid,
                    "trap_type": base_meta.get("trap_type", "") if base_meta else "",
                    "category": base_meta.get("category", "") if base_meta else "",
                    "metadata": base_meta,
                    "scores": scores,
                }
            )

        return {
            "configs": configs,
            "traps_meta": traps_meta,
            "results": results,
        }
