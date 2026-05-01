import json
import os
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
                "avg_faithfulness": sum(h.faithfulness_score for h in self.hallucination_steps)
                / len(self.hallucination_steps),
                "labels": [h.gsar_label for h in self.hallucination_steps],
            }
        if self.code_agent_checks:
            result["code_hallu"] = {
                "count": len(self.code_agent_checks),
                "types": [c.hallucination_type for c in self.code_agent_checks],
            }
        return result


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
        agent = agent_type or self.config.agent_type
        sandbox_type = self.config.sandbox.lower()

        if agent == "langchain":
            from agent_trust_lab.adapters.harnesses import LangChainHarness

            return LangChainHarness(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

        if agent == "openai":
            from agent_trust_lab.adapters.harnesses import OpenAIFunctionHarness

            return OpenAIFunctionHarness(model=self.config.model)

        if agent == "codex":
            from agent_trust_lab.adapters.harnesses import CodexHarness

            return CodexHarness(
                model=self.config.model,
                codebase_path=self.config.codebase_path,
            )

        if sandbox_type == "dry-run":
            from agent_trust_lab.sandbox.backends import DryRunSandbox

            return DryRunSandbox()

        if sandbox_type == "docker":
            from agent_trust_lab.sandbox.backends import DockerSandbox

            return DockerSandbox(
                image=self.config.sandbox_image,
                timeout=self.config.timeout,
                network_enabled=self.config.sandbox_network,
                tmpfs_size=self.config.sandbox_tmpfs_size,
                docker_host=self.config.docker_host,
            )

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

        hallu_steps, code_hallus = self._run_hallukg(
            trajectory=trajectory,
            is_code_agent=trap.category == "code_agent",
        )

        return EvaluationResult(
            trap_id=trap.trap_id,
            trap_type=trap.trap_type,
            category=trap.category,
            trajectory=trajectory,
            mutated=mutate,
            mutation_seed=mutation_seed if mutate else None,
            metadata={
                "severity": trap.severity,
                "difficulty": trap.difficulty,
            },
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
    ) -> tuple[List[HalluStepReport], List[CodeHalluReport]]:
        from agent_trust_lab.hallukg.anchoring import AnchoringReasoner
        from agent_trust_lab.hallukg.classifier import GSARClassifier
        from agent_trust_lab.hallukg.extractor import TripleExtractor

        extractor = TripleExtractor(model_name=self.config.model)
        reasoner = AnchoringReasoner(knowledge_base_path=self.config.anchor_kb)
        classifier = GSARClassifier()

        all_triples = []
        for step in trajectory.steps:
            if step.type == "trap_injection":
                continue
            triples = extractor.extract(step.content)
            anchored = reasoner.batch_anchor(triples)
            all_triples.extend(anchored)

        hallucination_steps = classifier.classify(trajectory.steps, all_triples)
        code_hallus: List[CodeHalluReport] = []

        if is_code_agent:
            from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

            code_checker = CodeHalluChecker(timeout=self.config.timeout)
            code_hallus = code_checker.batch_check(trajectory)

        return hallucination_steps, code_hallus

    def run_traps(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        mutate: bool = False,
        mutation_seed: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[EvaluationResult]:
        traps = self.trap_manager.load_traps(
            trap_ids=trap_ids,
            category=category,
            difficulty=difficulty,
            include_controls=True,
        )

        if limit:
            traps = traps[:limit]

        harness = self.resolve_harness()
        results: List[EvaluationResult] = []

        for trap in traps:
            result = self.run_single(
                trap, harness=harness, mutate=mutate, mutation_seed=mutation_seed
            )
            results.append(result)

        return results

    def export_results(self, results: List[EvaluationResult], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        payload = {
            "config": {
                "agent_type": self.config.agent_type,
                "model": self.config.model,
                "sandbox": self.config.sandbox,
            },
            "results": [r.summary() for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
