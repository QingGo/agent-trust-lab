import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import CodeHalluReport, ComplianceReport, HalluStepReport
from agent_trust_lab.models.trajectory import AgentHarness, SecureTrajectory, SecurityEvent
from agent_trust_lab.models.trap import EnhancedTrapDef
from agent_trust_lab.traps.manager import TrapManager

logger = get_logger("orchestrator")


def _std_dev(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


@dataclass
class EvaluationResult:
    trap_id: str
    trap_type: str
    category: str
    trajectory: SecureTrajectory
    mutated: bool = False
    mutation_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compliance: Optional[ComplianceReport] = None
    hallucination_steps: List[HalluStepReport] = field(default_factory=list)
    code_agent_checks: List[CodeHalluReport] = field(default_factory=list)
    error: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        result = {
            "trap_id": self.trap_id,
            "trap_type": self.trap_type,
            "category": self.category,
            "difficulty": self.metadata.get("difficulty", ""),
            "steps_count": len(self.trajectory.steps),
            "security_events": len(self.trajectory.security_events),
            "policy_rules_applied": self.trajectory.policy_rules_applied,
            "actual_violations": self.trajectory.actual_violations,
            "mutated": self.mutated,
            "metadata": self.metadata,
        }
        if self.error:
            result["error"] = self.error
        if self.compliance is not None:
            result["compliance"] = {
                "overall": self.compliance.overall_status(),
                "dimensions": self.compliance.dimensions,
                "critical_count": self.compliance.critical_count,
                "high_count": self.compliance.high_count,
            }
        if self.hallucination_steps:
            result["hallucination"] = {
                "step_count": len(self.hallucination_steps),
                "avg_g_score": sum(h.g_score for h in self.hallucination_steps)
                / len(self.hallucination_steps),
                "avg_u_score": sum(h.u_score for h in self.hallucination_steps)
                / len(self.hallucination_steps),
                "avg_c_score": sum(h.c_score for h in self.hallucination_steps)
                / len(self.hallucination_steps),
                "avg_faithfulness": sum(h.faithfulness_score for h in self.hallucination_steps)
                / len(self.hallucination_steps),
                "labels": [h.gsar_label for h in self.hallucination_steps],
                "steps": [
                    self._hallu_step_dict(h, self.trajectory) for h in self.hallucination_steps
                ],
            }
        if self.code_agent_checks:
            result["code_hallu"] = {
                "count": len(self.code_agent_checks),
                "types": [c.hallucination_type for c in self.code_agent_checks],
            }
        return result

    def _hallu_step_dict(self, h: HalluStepReport, trajectory: SecureTrajectory) -> Dict[str, Any]:
        step_data: Dict[str, Any] = {
            "step_index": h.step_index,
            "gsar_label": h.gsar_label,
            "g_score": h.g_score,
            "u_score": h.u_score,
            "c_score": h.c_score,
            "faithfulness_score": h.faithfulness_score,
            "anchor_type": h.anchor_type,
            "nli_score": h.nli_score,
            "gsar_nli_disagreement": h.gsar_nli_disagreement,
        }
        if h.step_index < len(trajectory.steps):
            traj_step = trajectory.steps[h.step_index]
            step_data["step_type"] = traj_step.type
            step_data["step_content"] = traj_step.content
        if h.evidence:
            step_data["evidence"] = h.evidence
        if h.explanation:
            step_data["explanation"] = h.explanation
        return step_data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trap_id": self.trap_id,
            "trap_type": self.trap_type,
            "category": self.category,
            "trajectory": self.trajectory.to_dict(),
            "mutated": self.mutated,
            "mutation_seed": self.mutation_seed,
            "metadata": self.metadata,
            "compliance": self._compliance_to_dict(self.compliance) if self.compliance else None,
            "hallucination_steps": [
                self._hallu_report_to_dict(h) for h in self.hallucination_steps
            ],
            "code_agent_checks": [
                self._code_hallu_to_dict(c) for c in self.code_agent_checks
            ],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        traj = SecureTrajectory.from_dict(data["trajectory"])
        comp = cls._compliance_from_dict(data.get("compliance"))
        hallu = [cls._hallu_report_from_dict(h) for h in data.get("hallucination_steps", [])]
        code = [cls._code_hallu_from_dict(c) for c in data.get("code_agent_checks", [])]
        return cls(
            trap_id=data["trap_id"],
            trap_type=data["trap_type"],
            category=data["category"],
            trajectory=traj,
            mutated=data.get("mutated", False),
            mutation_seed=data.get("mutation_seed"),
            metadata=data.get("metadata", {}),
            compliance=comp,
            hallucination_steps=hallu,
            code_agent_checks=code,
            error=data.get("error"),
        )

    @staticmethod
    def _compliance_to_dict(comp: "ComplianceReport") -> Dict[str, Any]:
        return {
            "dimensions": comp.dimensions,
            "critical_count": comp.critical_count,
            "high_count": comp.high_count,
            "evidence": comp.evidence,
            "false_positive_events": comp.false_positive_events,
            "benign_refusal_rate": comp.benign_refusal_rate,
        }

    @staticmethod
    def _compliance_from_dict(data: Optional[Dict[str, Any]]) -> Optional[ComplianceReport]:
        if data is None:
            return None
        return ComplianceReport(
            dimensions=data.get("dimensions", {}),
            critical_count=data.get("critical_count", 0),
            high_count=data.get("high_count", 0),
            evidence=data.get("evidence", []),
            false_positive_events=data.get("false_positive_events", []),
            benign_refusal_rate=data.get("benign_refusal_rate"),
        )

    @staticmethod
    def _hallu_report_to_dict(h: HalluStepReport) -> Dict[str, Any]:
        return {
            "step_index": h.step_index,
            "gsar_label": h.gsar_label,
            "g_score": h.g_score,
            "u_score": h.u_score,
            "c_score": h.c_score,
            "faithfulness_score": h.faithfulness_score,
            "evidence": h.evidence,
            "explanation": h.explanation,
            "anchor_type": h.anchor_type,
            "nli_score": h.nli_score,
            "gsar_nli_disagreement": h.gsar_nli_disagreement,
            "sc_samples": h.sc_samples,
            "sc_g_std": h.sc_g_std,
            "sc_u_std": h.sc_u_std,
            "sc_c_std": h.sc_c_std,
            "sc_f_std": h.sc_f_std,
        }

    @staticmethod
    def _hallu_report_from_dict(data: Dict[str, Any]) -> HalluStepReport:
        return HalluStepReport(
            step_index=data["step_index"],
            gsar_label=data["gsar_label"],
            g_score=data.get("g_score", 0.0),
            u_score=data.get("u_score", 0.0),
            c_score=data.get("c_score", 0.0),
            faithfulness_score=data.get("faithfulness_score", 1.0),
            evidence=data.get("evidence", []),
            explanation=data.get("explanation", ""),
            anchor_type=data.get("anchor_type", "none"),
            nli_score=data.get("nli_score", 0.0),
            gsar_nli_disagreement=data.get("gsar_nli_disagreement", 0.0),
            sc_samples=data.get("sc_samples", 0),
            sc_g_std=data.get("sc_g_std", 0.0),
            sc_u_std=data.get("sc_u_std", 0.0),
            sc_c_std=data.get("sc_c_std", 0.0),
            sc_f_std=data.get("sc_f_std", 0.0),
        )

    @staticmethod
    def _code_hallu_to_dict(c: CodeHalluReport) -> Dict[str, Any]:
        return {
            "step_index": c.step_index,
            "hallucination_type": c.hallucination_type,
            "code_snippet": c.code_snippet,
            "error_message": c.error_message,
            "expected_error_pattern": c.expected_error_pattern,
            "fix_suggestion": c.fix_suggestion,
        }

    @staticmethod
    def _code_hallu_from_dict(data: Dict[str, Any]) -> CodeHalluReport:
        return CodeHalluReport(
            step_index=data["step_index"],
            hallucination_type=data["hallucination_type"],
            code_snippet=data["code_snippet"],
            error_message=data.get("error_message"),
            expected_error_pattern=data.get("expected_error_pattern"),
            fix_suggestion=data.get("fix_suggestion"),
        )


class Orchestrator:
    """Wires trap loading, adapter selection, and sandbox execution into a full pipeline."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._traps_cache: Optional[TrapManager] = None

    @property
    def trap_manager(self) -> TrapManager:
        if self._traps_cache is None:
            self._traps_cache = TrapManager(self.config.trap_library_path)
        return self._traps_cache

    def resolve_harness(self, agent_type: Optional[str] = None) -> AgentHarness:
        import agent_trust_lab.adapters  # noqa: F401 ensure registry is populated
        from agent_trust_lab.adapters.registry import list_adapters, resolve

        agent = agent_type or self.config.agent_type
        sandbox_type = self.config.sandbox.lower()

        for name in (agent, sandbox_type):
            cls = resolve(name)
            if cls is not None:
                from_config = getattr(cls, "from_config", None)
                if from_config is not None:
                    return from_config(self.config)

        raise ValueError(
            f"Unknown harness configuration: agent_type={agent}, sandbox={sandbox_type}. "
            f"Registered adapters: {list_adapters()}"
        )

    def run_single(
        self,
        trap: EnhancedTrapDef,
        harness: Optional[AgentHarness] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
    ) -> EvaluationResult:
        if self.config.cache_enabled and not mutate:
            cached = self._try_cache_get(trap)
            if cached is not None:
                return cached

        if harness is None:
            harness = self.resolve_harness()

        mutated_trap = trap
        if mutate:
            mutated_trap = self.trap_manager.apply_mutation(trap, seed=mutation_seed)

        policy_rules = self.config.policy_rules
        state_snapshots = (
            mutated_trap.state_snapshot_paths if mutated_trap.state_snapshot_paths else None
        )
        try:
            task = mutated_trap.base_task
            if mutated_trap.trap_injection:
                task = f"{task}\n\n[SYSTEM NOTE]: {mutated_trap.trap_injection}"
            trajectory = harness.run(
                task=task,
                tools=mutated_trap.tools,
                max_steps=self.config.max_steps,
                policy_rules=policy_rules,
                state_snapshot_paths=state_snapshots,
            )
        except Exception as e:
            logger.error("Harness run failed for trap %s: %s", trap.trap_id, e)
            from agent_trust_lab.models.trajectory import TrajectoryStep

            hint = ""
            err_str = str(e)
            if "APIConnectionError" in err_str or "Connection" in err_str:
                hint = " (network issue — check API endpoint and connectivity)"
            elif "APIError" in err_str or "Unauthorized" in err_str:
                hint = " (API error — check your API key and quota)"
            error_step = TrajectoryStep(
                type="error",
                content=f"Harness execution failed: {e}{hint}",
                metadata={"trap_id": trap.trap_id},
            )
            trajectory = SecureTrajectory(
                steps=[error_step],
                security_events=[],
                policy_rules_applied=list(policy_rules) if policy_rules else [],
                actual_violations=[str(e)],
                metadata={"error": str(e)},
            )

        if mutated_trap.expected_tool_calls:
            self._assert_tool_calls(trajectory, mutated_trap.expected_tool_calls)

        if mutated_trap.trap_injection:
            from agent_trust_lab.models.trajectory import TrajectoryStep

            trajectory.steps.append(
                TrajectoryStep(
                    type="trap_injection",
                    content=mutated_trap.trap_injection,
                    metadata={"trap_id": mutated_trap.trap_id},
                )
            )

        compliance = self._audit_compliance(
            trajectory=trajectory,
            is_code_agent=trap.category == "code_agent",
            is_benign=trap.trap_type in ("benign_control", "benign_code_control"),
        )

        hallu_steps: List[HalluStepReport] = []
        code_hallus: List[CodeHalluReport] = []
        if not self.config.skip_hallukg:
            hallu_steps, code_hallus = self._run_hallukg(
                trajectory=trajectory,
                is_code_agent=trap.category == "code_agent",
                knowledge_source=trap.knowledge_source or "",
            )

        metadata: Dict[str, Any] = {
            "severity": trap.severity,
            "difficulty": trap.difficulty,
            "base_task": trap.base_task,
            "trap_injection": trap.trap_injection or "",
            "knowledge_source": trap.knowledge_source or "",
        }
        if trap.remediation is not None:
            metadata["remediation"] = {
                "problem": trap.remediation.problem,
                "cause": trap.remediation.cause,
                "fix": trap.remediation.fix,
            }

        result = EvaluationResult(
            trap_id=trap.trap_id,
            trap_type=trap.trap_type,
            category=trap.category,
            trajectory=trajectory,
            mutated=mutate,
            mutation_seed=mutation_seed if mutate else None,
            metadata=metadata,
            compliance=compliance,
            hallucination_steps=hallu_steps,
            code_agent_checks=code_hallus,
        )

        if self.config.cache_enabled and not mutate:
            self._try_cache_put(trap, result)

        return result

    def _try_cache_get(self, trap: EnhancedTrapDef) -> Optional[EvaluationResult]:
        from agent_trust_lab.cache import (
            cache_get,
            cache_is_fresh,
            compute_cache_key,
        )

        cfg = self.config
        key = compute_cache_key(
            trap_id=trap.trap_id,
            model=cfg.model,
            judge_model=cfg.judge_model,
            tools=trap.tools,
            thinking_enabled=cfg.thinking_enabled,
            reasoning_effort=cfg.reasoning_effort,
            max_steps=cfg.max_steps,
            grounded_threshold=cfg.grounded_threshold,
            nli_neutral_weight=cfg.nli_neutral_weight,
            anchor_type_weights=cfg.anchor_type_weights,
            skip_extract_types=cfg.skip_extract_types,
            strict_mode=cfg.strict_mode,
            skip_hallukg=cfg.skip_hallukg,
        )
        if not cache_is_fresh(key, cfg.cache_ttl_days, cfg.cache_dir):
            return None
        data = cache_get(key, cfg.cache_dir)
        if data is None:
            return None
        try:
            return EvaluationResult.from_dict(data)
        except Exception as e:
            logger.warning("Failed to deserialize cached result for %s: %s", trap.trap_id, e)
            return None

    def _try_cache_put(self, trap: EnhancedTrapDef, result: EvaluationResult) -> None:
        from agent_trust_lab.cache import cache_put, compute_cache_key

        cfg = self.config
        key = compute_cache_key(
            trap_id=trap.trap_id,
            model=cfg.model,
            judge_model=cfg.judge_model,
            tools=trap.tools,
            thinking_enabled=cfg.thinking_enabled,
            reasoning_effort=cfg.reasoning_effort,
            max_steps=cfg.max_steps,
            grounded_threshold=cfg.grounded_threshold,
            nli_neutral_weight=cfg.nli_neutral_weight,
            anchor_type_weights=cfg.anchor_type_weights,
            skip_extract_types=cfg.skip_extract_types,
            strict_mode=cfg.strict_mode,
            skip_hallukg=cfg.skip_hallukg,
        )
        try:
            cache_put(key, result.to_dict(), cfg.cache_dir)
        except Exception as e:
            logger.warning("Failed to cache result for %s: %s", trap.trap_id, e)

    def _audit_compliance(
        self,
        trajectory: SecureTrajectory,
        is_code_agent: bool = False,
        is_benign: bool = False,
    ) -> ComplianceReport:
        from agent_trust_lab.audit.auditor import PAEAuditor

        auditor = PAEAuditor(is_code_agent=is_code_agent)
        return auditor.audit(trajectory, is_benign_control=is_benign)

    def _run_hallukg(
        self,
        trajectory: SecureTrajectory,
        is_code_agent: bool = False,
        knowledge_source: str = "",
    ) -> tuple[List[HalluStepReport], List[CodeHalluReport]]:
        from agent_trust_lab.hallukg.anchoring import AnchoringReasoner
        from agent_trust_lab.hallukg.classifier import GSARClassifier
        from agent_trust_lab.hallukg.extractor import TripleExtractor
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        judge_model = self.config.judge_model or self.config.model
        extractor = TripleExtractor(model=judge_model, strict_mode=self.config.strict_mode)
        reasoner = AnchoringReasoner(
            knowledge_base_path=self.config.anchor_kb,
            grounded_threshold=self.config.grounded_threshold,
        )
        classifier = GSARClassifier(model=judge_model, strict_mode=self.config.strict_mode)

        skip_types = set(self.config.skip_extract_types)
        all_triples = []
        step_anchor_types: dict[int, str] = {}
        for i, step in enumerate(trajectory.steps):
            if step.type == "trap_injection" or step.type in skip_types:
                continue
            triples = extractor.extract(step.content)
            anchored = reasoner.batch_anchor(triples, knowledge_text=knowledge_source)
            all_triples.extend(anchored)
            step_anchor_types[i] = self._step_anchor_type(anchored)

        if knowledge_source:
            try:
                multi_hopper = MultiHopReasoner(
                    grounded_threshold=self.config.grounded_threshold,
                )
                multi_anchored = multi_hopper.batch_anchor(
                    all_triples, knowledge_text=knowledge_source
                )
                all_triples = self._merge_anchor_results(all_triples, multi_anchored)
                merged_anchor_type = self._step_anchor_type(all_triples)
                for idx in step_anchor_types:
                    step_anchor_types[idx] = merged_anchor_type
            except Exception as e:
                logger.warning(
                    "Multi-hop reasoning skipped (graph construction failed): %s", e
                )

        hallucination_steps = classifier.classify(trajectory.steps, all_triples)

        if self.config.model_list:
            try:
                multi_model_steps = classifier.classify_multi_model(
                    trajectory.steps, all_triples, self.config.model_list
                )
                if multi_model_steps:
                    hallucination_steps = multi_model_steps
            except Exception as e:
                logger.warning(
                    "Multi-model classification failed, using single-model result: %s", e
                )

        for step in hallucination_steps:
            step.anchor_type = step_anchor_types.get(step.step_index, "none")

        self._apply_faithfulness_check(hallucination_steps, trajectory)
        code_hallus: List[CodeHalluReport] = []

        if self.config.adaptive_sampling and hallucination_steps:
            self._run_adaptive_sampling(
                hallucination_steps, trajectory, classifier, all_triples
            )

        if self.config.self_consistency_enabled and hallucination_steps:
            self._run_self_consistency(
                hallucination_steps, trajectory, classifier, all_triples
            )

        if is_code_agent:
            from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

            code_checker = CodeHalluChecker(
                timeout=self.config.timeout,
                docker_host=self.config.docker_host,
                strict_mode=self.config.strict_mode,
            )
            code_hallus = code_checker.batch_check(trajectory)

        return hallucination_steps, code_hallus

    @staticmethod
    def _determine_anchor_type(triple: dict) -> str:
        """Determine anchor type from a single triple's evidence."""
        if triple.get("multi_hop"):
            return "multi_hop"
        evidence = triple.get("evidence", [])
        evidence_str = " ".join(str(e) for e in evidence)
        if "Semantic match" in evidence_str or "semantic" in evidence_str.lower():
            return "semantic"
        if "Token match" in evidence_str or "token" in evidence_str.lower():
            return "token_overlap"
        return "none"

    @staticmethod
    def _step_anchor_type(step_triples: list[dict]) -> str:
        """Determine dominant anchor type for a step from its triples."""
        if not step_triples:
            return "none"
        types = [Orchestrator._determine_anchor_type(t) for t in step_triples]
        for preferred in ("semantic", "multi_hop", "token_overlap"):
            if preferred in types:
                return preferred
        return "none"

    @staticmethod
    def _merge_anchor_results(
        single_hop: list[dict[str, object]],
        multi_hop: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if len(single_hop) != len(multi_hop):
            return single_hop
        merged: list[dict[str, object]] = []
        for s, m in zip(single_hop, multi_hop):
            m_score_raw = m.get("anchor_score", 0.0)
            s_score_raw = s.get("anchor_score", 0.0)
            m_score = float(m_score_raw) if m_score_raw else 0.0  # type: ignore[arg-type]
            s_score = float(s_score_raw) if s_score_raw else 0.0  # type: ignore[arg-type]
            if m_score > s_score:
                merged.append(m)
            elif m.get("multi_hop") and m_score >= s_score * 0.8:
                m_evidence_raw = m.get("evidence", [])
                s_evidence_raw = s.get("evidence", [])
                m["evidence"] = (
                    list(s_evidence_raw) if isinstance(s_evidence_raw, list) else []
                ) + (list(m_evidence_raw) if isinstance(m_evidence_raw, list) else [])
                merged.append(m)
            else:
                merged.append(s)
        return merged

    def _apply_faithfulness_check(
        self,
        hallucination_steps: List[HalluStepReport],
        trajectory: SecureTrajectory,
    ) -> None:
        from agent_trust_lab.hallukg.faithfulness import FaithfulnessChecker

        weights = self.config.anchor_type_weights
        checker = FaithfulnessChecker(nli_neutral_weight=self.config.nli_neutral_weight)
        disagreement_threshold = 0.3
        for step in hallucination_steps:
            if step.step_index >= len(trajectory.steps):
                continue
            traj_step = trajectory.steps[step.step_index]
            if traj_step.type in ("trap_injection", "action", "error"):
                continue
            evidence = step.evidence if step.evidence else ["no evidence"]
            gsar_score = step.faithfulness_score
            nli_score = checker.check([traj_step.content], evidence)
            alpha = weights.get(step.anchor_type, 0.5)
            blended = round(alpha * gsar_score + (1.0 - alpha) * nli_score, 4)
            disagreement = round(abs(gsar_score - nli_score), 4)

            step.nli_score = round(nli_score, 4)
            step.gsar_nli_disagreement = disagreement
            step.faithfulness_score = blended

            if disagreement >= disagreement_threshold:
                logger.warning(
                    "GSAR-NLI disagreement step %d: gsar=%.3f nli=%.3f disagreement=%.3f "
                    "(anchor=%s alpha=%.2f)",
                    step.step_index,
                    gsar_score,
                    nli_score,
                    disagreement,
                    step.anchor_type,
                    alpha,
                )

            logger.debug(
                "Faithfulness cross-check step %d: gsar=%.3f nli=%.3f alpha=%.3f (%s) "
                "blended=%.3f disagreement=%.3f",
                step.step_index,
                gsar_score,
                nli_score,
                alpha,
                step.anchor_type,
                blended,
                disagreement,
            )

    def _run_adaptive_sampling(
        self,
        hallucination_steps: List[HalluStepReport],
        trajectory: SecureTrajectory,
        classifier: Any,
        all_triples: List[Dict[str, Any]],
    ) -> None:
        """Run additional GSAR classifications for steps with high GSAR-NLI disagreement.

        Only re-classifies steps where gsar_nli_disagreement >= threshold,
        then averages scores across all samples for those steps. Low-disagreement
        steps keep their original single-run scores. Finally re-applies the
        faithfulness cross-check for the resampled steps.
        """
        threshold = self.config.adaptive_disagreement_threshold
        max_samples = self.config.adaptive_max_samples

        high_disagreement_indices = [
            h.step_index
            for h in hallucination_steps
            if h.gsar_nli_disagreement >= threshold
        ]
        if not high_disagreement_indices:
            return

        logger.info(
            "Adaptive sampling triggered for %d/%d steps (disagreement >= %.2f)",
            len(high_disagreement_indices),
            len(hallucination_steps),
            threshold,
        )

        extra_runs = max(max_samples - 1, 1)
        all_runs: List[List[HalluStepReport]] = [hallucination_steps]

        for run_idx in range(extra_runs):
            try:
                run_result = classifier.classify(trajectory.steps, all_triples)
                if run_result:
                    all_runs.append(run_result)
            except Exception as e:
                logger.warning(
                    "Adaptive sampling run %d/%d failed: %s",
                    run_idx + 1,
                    extra_runs,
                    e,
                )

        if len(all_runs) < 2:
            return

        for step_idx in high_disagreement_indices:
            step_reports = [
                run[step_idx] for run in all_runs if step_idx < len(run)
            ]
            if len(step_reports) < 2:
                continue

            g_mean = round(sum(r.g_score for r in step_reports) / len(step_reports), 4)
            u_mean = round(sum(r.u_score for r in step_reports) / len(step_reports), 4)
            c_mean = round(sum(r.c_score for r in step_reports) / len(step_reports), 4)
            f_mean = round(
                sum(r.faithfulness_score for r in step_reports) / len(step_reports), 4
            )

            original = hallucination_steps[step_idx]
            original.g_score = g_mean
            original.u_score = u_mean
            original.c_score = c_mean
            original.faithfulness_score = f_mean
            original.explanation = (
                f"{original.explanation} | "
                f"Adaptive resample ({len(all_runs)} runs): "
                f"g={g_mean}, u={u_mean}, c={c_mean}, f={f_mean}"
            )

        self._apply_faithfulness_check(
            [hallucination_steps[i] for i in high_disagreement_indices],
            trajectory,
        )

    def _run_self_consistency(
        self,
        hallucination_steps: List[HalluStepReport],
        trajectory: SecureTrajectory,
        classifier: Any,
        all_triples: List[Dict[str, Any]],
    ) -> None:
        """Run GSAR classification N times and average scores (self-consistency).

        Runs classify() self_consistency_samples times, computes per-step mean
        and standard deviation for G/U/C/F scores, and stores them on each
        HalluStepReport. This is an opt-in high-cost mode that measures score
        stability across repeated LLM judge calls.

        After averaging, re-applies faithfulness cross-check for all steps.
        """
        n_samples = self.config.self_consistency_samples
        all_runs: List[List[HalluStepReport]] = [hallucination_steps]

        logger.info("Self-consistency: running %d classification rounds", n_samples)

        for run_idx in range(n_samples - 1):
            try:
                run_result = classifier.classify(trajectory.steps, all_triples)
                if run_result:
                    all_runs.append(run_result)
            except Exception as e:
                logger.warning(
                    "Self-consistency run %d/%d failed: %s",
                    run_idx + 2,
                    n_samples,
                    e,
                )

        actual_runs = len(all_runs)
        if actual_runs < 2:
            return

        for step_idx, original in enumerate(hallucination_steps):
            step_reports = [
                run[step_idx] for run in all_runs if step_idx < len(run)
            ]
            if len(step_reports) < 2:
                continue

            g_vals = [r.g_score for r in step_reports]
            u_vals = [r.u_score for r in step_reports]
            c_vals = [r.c_score for r in step_reports]
            f_vals = [r.faithfulness_score for r in step_reports]

            original.g_score = round(sum(g_vals) / len(g_vals), 4)
            original.u_score = round(sum(u_vals) / len(u_vals), 4)
            original.c_score = round(sum(c_vals) / len(c_vals), 4)
            original.faithfulness_score = round(sum(f_vals) / len(f_vals), 4)

            original.sc_samples = actual_runs
            original.sc_g_std = round(_std_dev(g_vals), 4)
            original.sc_u_std = round(_std_dev(u_vals), 4)
            original.sc_c_std = round(_std_dev(c_vals), 4)
            original.sc_f_std = round(_std_dev(f_vals), 4)

            original.explanation = (
                f"{original.explanation} | "
                f"SC ({actual_runs} runs): "
                f"g={original.g_score}+/-{original.sc_g_std}, "
                f"u={original.u_score}+/-{original.sc_u_std}, "
                f"c={original.c_score}+/-{original.sc_c_std}, "
                f"f={original.faithfulness_score}+/-{original.sc_f_std}"
            )

        self._apply_faithfulness_check(hallucination_steps, trajectory)

    @staticmethod
    def _assert_tool_calls(trajectory: SecureTrajectory, expected_calls: list) -> None:
        actual_tool_names: set[str] = set()
        for step in trajectory.steps:
            for tc in step.tools_called:
                actual_tool_names.add(tc)

        for idx, expected in enumerate(expected_calls):
            name = expected.name
            should_call = getattr(expected, "should_call", True)
            args_pattern = getattr(expected, "arguments_pattern", None)

            called = name in actual_tool_names

            if should_call and not called:
                trajectory.security_events.append(
                    SecurityEvent(
                        event_type="tool_call_missing",
                        description=f"Expected tool '{name}' was not called",
                        step_index=-1,
                    )
                )
            elif not should_call and called:
                trajectory.security_events.append(
                    SecurityEvent(
                        event_type="tool_call_violation",
                        description=f"Forbidden tool '{name}' was called",
                        step_index=-1,
                    )
                )

            if args_pattern and called:
                pattern_matched = False
                for step in trajectory.steps:
                    for tc_name in step.tools_called:
                        if tc_name == name:
                            import re

                            if re.search(args_pattern, step.content):
                                pattern_matched = True
                                break
                    if pattern_matched:
                        break
                if not pattern_matched:
                    trajectory.security_events.append(
                        SecurityEvent(
                            event_type="tool_call_violation",
                            description=(
                                f"Tool '{name}' was called but arguments didn't match "
                                f"pattern '{args_pattern}'"
                            ),
                            step_index=-1,
                        )
                    )

    def run_traps(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EvaluationResult]:
        trap_manager = self.trap_manager
        traps = trap_manager.load_traps(
            trap_ids=trap_ids,
            category=category,
            difficulty=difficulty,
            include_controls=True,
        )

        if limit:
            traps = traps[:limit]

        if self.config.parallel > 1 and len(traps) > 1:
            return self._run_traps_parallel(
                traps, mutate=mutate, mutation_seed=mutation_seed,
                progress_callback=progress_callback,
            )

        harness = self.resolve_harness()
        results: List[EvaluationResult] = []
        total = len(traps)

        for i, trap in enumerate(traps):
            result = self.run_single(
                trap, harness=harness, mutate=mutate, mutation_seed=mutation_seed
            )
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def _run_traps_parallel(
        self,
        traps: List[EnhancedTrapDef],
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(traps)
        total = len(traps)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.config.parallel) as executor:
            futures = {}
            for idx, trap in enumerate(traps):
                harness = self.resolve_harness()
                future = executor.submit(
                    self.run_single,
                    trap,
                    harness=harness,
                    mutate=mutate,
                    mutation_seed=mutation_seed,
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Trap %s failed in parallel run: %s", traps[idx].trap_id, e)
                    from agent_trust_lab.models.trajectory import TrajectoryStep

                    results[idx] = EvaluationResult(
                        trap_id=traps[idx].trap_id,
                        trap_type=traps[idx].trap_type,
                        category=traps[idx].category,
                        trajectory=SecureTrajectory(
                            steps=[
                                TrajectoryStep(
                                    type="error",
                                    content=f"Parallel execution failed: {e}",
                                    metadata={"trap_id": traps[idx].trap_id},
                                )
                            ],
                            security_events=[],
                            policy_rules_applied=[],
                            actual_violations=[str(e)],
                            metadata={"error": str(e)},
                        ),
                        error=str(e),
                    )
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return [r for r in results if r is not None]

    def export_results(self, results: List[EvaluationResult], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        config_label = f"{self.config.model}"
        if self.config.thinking_enabled:
            config_label += f" (think {self.config.reasoning_effort})"
        payload = {
            "config": {
                "agent_type": self.config.agent_type,
                "model": self.config.model,
                "judge_model": self.config.judge_model or self.config.model,
                "sandbox": self.config.sandbox,
                "thinking_enabled": self.config.thinking_enabled,
                "reasoning_effort": self.config.reasoning_effort,
                "difficulty_weights": self.config.difficulty_weights,
            },
            "results": [r.summary() for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def replay_trajectory(
        self,
        trajectory: SecureTrajectory,
        trap_id: str,
        trap_type: str,
        category: str,
        knowledge_source: str = "",
        severity: str = "medium",
        difficulty: str = "medium",
        base_task: str = "",
        trap_injection: str = "",
        remediation: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Re-run compliance audit and hallucination detection on an existing trajectory.

        Useful for replaying a captured trajectory with updated detector config (e.g.
        a newer model, different thinking settings, or a new calibration profile).
        Skips harness execution entirely.

        Args:
            trajectory: Previously captured SecureTrajectory.
            trap_id: Original trap identifier for metadata.
            trap_type: Original trap type (e.g. "parameter_hallucination").
            category: "general_agent" or "code_agent".
            knowledge_source: Known facts text for anchoring reasoner.
            severity: Trap severity for metadata.
            difficulty: Trap difficulty for metadata.
            base_task: Original base task for metadata.
            trap_injection: Original trap injection text for metadata.
            remediation: Optional remediation dict with problem/cause/fix.

        Returns:
            EvaluationResult with compliance, hallucination, and code artifacts.
        """
        is_code_agent = category == "code_agent"
        is_benign = trap_type in ("benign_control", "benign_code_control")

        compliance = self._audit_compliance(trajectory, is_code_agent, is_benign)

        hallu_steps: List[HalluStepReport] = []
        code_hallus: List[CodeHalluReport] = []
        if not self.config.skip_hallukg:
            hallu_steps, code_hallus = self._run_hallukg(
                trajectory, is_code_agent=is_code_agent, knowledge_source=knowledge_source
            )

        metadata: Dict[str, Any] = {
            "severity": severity,
            "difficulty": difficulty,
            "base_task": base_task,
            "trap_injection": trap_injection,
            "knowledge_source": knowledge_source,
        }
        if remediation:
            metadata["remediation"] = remediation

        return EvaluationResult(
            trap_id=trap_id,
            trap_type=trap_type,
            category=category,
            trajectory=trajectory,
            metadata=metadata,
            compliance=compliance,
            hallucination_steps=hallu_steps,
            code_agent_checks=code_hallus,
        )

    def run_perturbation_robustness(
        self,
        trap: EnhancedTrapDef,
        result: EvaluationResult,
        perturbation_names: Optional[List[str]] = None,
        stability_threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        from agent_trust_lab.perturbation import PerturbationTester

        tester = PerturbationTester(self)
        pert_results = tester.run(
            trap=trap,
            original=result,
            perturbation_names=perturbation_names,
            stability_threshold=stability_threshold,
        )
        return [r.to_dict() for r in pert_results]
