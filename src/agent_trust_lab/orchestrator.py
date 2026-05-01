import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.models.trajectory import AgentHarness, SecureTrajectory
from agent_trust_lab.models.trap import EnhancedTrapDef
from agent_trust_lab.traps.manager import TrapManager


@dataclass
class EvaluationResult:
    trap_id: str
    trap_type: str
    category: str
    trajectory: SecureTrajectory
    mutated: bool = False
    mutation_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "trap_id": self.trap_id,
            "trap_type": self.trap_type,
            "category": self.category,
            "steps_count": len(self.trajectory.steps),
            "security_events": len(self.trajectory.security_events),
            "policy_violations": self.trajectory.policy_violations,
            "mutated": self.mutated,
            "metadata": self.metadata,
        }


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

        if sandbox_type == "dry-run":
            from agent_trust_lab.sandbox.backends import DryRunSandbox

            return DryRunSandbox()

        if sandbox_type == "docker":
            from agent_trust_lab.sandbox.backends import DockerSandbox

            return DockerSandbox(timeout=self.config.timeout)

        if agent == "langchain":
            from agent_trust_lab.adapters.harnesses import LangChainHarness

            return LangChainHarness(model=self.config.model)

        if agent == "openai":
            from agent_trust_lab.adapters.harnesses import OpenAIFunctionHarness

            return OpenAIFunctionHarness(model=self.config.model)

        if agent == "codex":
            from agent_trust_lab.adapters.harnesses import CodexHarness

            return CodexHarness(
                model=self.config.model,
                codebase_path=self.config.codebase_path,
            )

        from agent_trust_lab.sandbox.backends import DockerSandbox

        return DockerSandbox(timeout=self.config.timeout)

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
        trajectory = harness.run(
            task=mutated_trap.base_task,
            tools=mutated_trap.tools,
            max_steps=self.config.max_steps,
            policy_rules=policy_rules,
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
