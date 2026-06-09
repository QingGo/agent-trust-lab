import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_trust_lab.models.report import CodeHalluReport, ComplianceReport, HalluStepReport
from agent_trust_lab.models.trajectory import SecureTrajectory


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
    runs_count: int = 1
    run_details: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Dict[str, Any] = field(default_factory=dict)

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
        if self.runs_count > 1:
            result["runs_count"] = self.runs_count
            result["run_details"] = self.run_details
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
        if self.checkpoint:
            result["checkpoint"] = self.checkpoint
        if self.trajectory.steps:
            result["trajectory_steps"] = [
                {"type": s.type, "content": s.content}
                for s in self.trajectory.steps
            ]
        if self.trajectory.security_events:
            result["security_event_log"] = [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "step_index": e.step_index,
                }
                for e in self.trajectory.security_events
            ]
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
            "raw_gsar_faithfulness": h.raw_gsar_faithfulness,
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
        result = {
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
        if self.runs_count > 1:
            result["runs_count"] = self.runs_count
            result["run_details"] = self.run_details
        return result

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
            runs_count=data.get("runs_count", 1),
            run_details=data.get("run_details", []),
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
