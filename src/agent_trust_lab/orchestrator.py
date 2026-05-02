import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import CodeHalluReport, ComplianceReport, HalluStepReport
from agent_trust_lab.models.trajectory import AgentHarness, SecureTrajectory
from agent_trust_lab.models.trap import EnhancedTrapDef
from agent_trust_lab.traps.manager import TrapManager

logger = get_logger("orchestrator")


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
                    self._hallu_step_dict(h, self.trajectory)
                    for h in self.hallucination_steps
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
        from agent_trust_lab.adapters.registry import resolve

        agent = agent_type or self.config.agent_type
        sandbox_type = self.config.sandbox.lower()

        for name in (agent, sandbox_type):
            cls = resolve(name)
            if cls is not None:
                from_config = getattr(cls, "from_config", None)
                if from_config is not None:
                    return from_config(self.config)

        raise ValueError(
            f"Unknown harness configuration: agent_type={agent}, sandbox={sandbox_type}"
        )

    def run_single(
        self,
        trap: EnhancedTrapDef,
        harness: Optional[AgentHarness] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
    ) -> EvaluationResult:
        if harness is None:
            harness = self.resolve_harness()

        mutated_trap = trap
        if mutate:
            mutated_trap = self.trap_manager.apply_mutation(trap, seed=mutation_seed)

        policy_rules = self.config.policy_rules
        try:
            trajectory = harness.run(
                task=mutated_trap.base_task,
                tools=mutated_trap.tools,
                max_steps=self.config.max_steps,
                policy_rules=policy_rules,
            )
        except Exception as e:
            logger.error("Harness run failed for trap %s: %s", trap.trap_id, e)
            from agent_trust_lab.models.trajectory import TrajectoryStep

            error_step = TrajectoryStep(
                type="error",
                content=f"Harness execution failed: {e}",
                metadata={"trap_id": trap.trap_id},
            )
            trajectory = SecureTrajectory(
                steps=[error_step],
                security_events=[],
                policy_rules_applied=list(policy_rules) if policy_rules else [],
                actual_violations=[str(e)],
                metadata={"error": str(e)},
            )

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

        return EvaluationResult(
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

        extractor = TripleExtractor(model=self.config.model)
        reasoner = AnchoringReasoner(
            knowledge_base_path=self.config.anchor_kb,
            grounded_threshold=self.config.grounded_threshold,
        )
        classifier = GSARClassifier(model=self.config.model)

        skip_types = set(self.config.skip_extract_types)
        all_triples = []
        for step in trajectory.steps:
            if step.type == "trap_injection" or step.type in skip_types:
                continue
            triples = extractor.extract(step.content)
            anchored = reasoner.batch_anchor(triples, knowledge_text=knowledge_source)
            all_triples.extend(anchored)

        hallucination_steps = classifier.classify(trajectory.steps, all_triples)
        self._apply_faithfulness_check(hallucination_steps, trajectory)
        code_hallus: List[CodeHalluReport] = []

        if is_code_agent:
            from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

            code_checker = CodeHalluChecker(
                timeout=self.config.timeout, docker_host=self.config.docker_host
            )
            code_hallus = code_checker.batch_check(trajectory)

        return hallucination_steps, code_hallus

    def _apply_faithfulness_check(
        self,
        hallucination_steps: List[HalluStepReport],
        trajectory: SecureTrajectory,
    ) -> None:
        from agent_trust_lab.hallukg.faithfulness import FaithfulnessChecker

        checker = FaithfulnessChecker(nli_neutral_weight=self.config.nli_neutral_weight)
        for step in hallucination_steps:
            if step.step_index >= len(trajectory.steps):
                continue
            traj_step = trajectory.steps[step.step_index]
            if traj_step.type in ("trap_injection", "action", "error"):
                continue
            evidence = step.evidence if step.evidence else ["no evidence"]
            gsar_score = step.faithfulness_score
            nli_score = checker.check([traj_step.content], evidence)
            step.faithfulness_score = round((gsar_score + nli_score) / 2.0, 4)
            logger.debug(
                "Faithfulness cross-check step %d: gsar=%.3f nli=%.3f blended=%.3f",
                step.step_index,
                gsar_score,
                nli_score,
                step.faithfulness_score,
            )

    def run_traps(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        limit: Optional[int] = None,
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
                traps, mutate=mutate, mutation_seed=mutation_seed
            )

        harness = self.resolve_harness()
        results: List[EvaluationResult] = []

        for trap in traps:
            result = self.run_single(
                trap, harness=harness, mutate=mutate, mutation_seed=mutation_seed
            )
            results.append(result)

        return results

    def _run_traps_parallel(
        self,
        traps: List[EnhancedTrapDef],
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
    ) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(traps)

        with ThreadPoolExecutor(max_workers=self.config.parallel) as executor:
            futures = {}
            for idx, trap in enumerate(traps):
                harness = self.resolve_harness()
                future = executor.submit(
                    self.run_single, trap,
                    harness=harness, mutate=mutate, mutation_seed=mutation_seed,
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
                "sandbox": self.config.sandbox,
                "thinking_enabled": self.config.thinking_enabled,
                "reasoning_effort": self.config.reasoning_effort,
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
