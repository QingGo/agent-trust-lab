from typing import Optional

from agent_trust_lab.audit.rules import DimensionStatus, RuleSet
from agent_trust_lab.models.report import ComplianceReport
from agent_trust_lab.models.trajectory import SecureTrajectory, SecurityEvent, TrajectoryStep


class PAEAuditor:
    """PAE Compliance Auditor: evaluates agent trajectory against six-dimension gating rules."""

    def __init__(self, is_code_agent: bool = False):
        self.rule_set = RuleSet()
        self.is_code_agent = is_code_agent

    def audit(
        self,
        trajectory: SecureTrajectory,
        is_benign_control: bool = False,
    ) -> ComplianceReport:
        dimensions: dict[str, str] = {}
        evidence: list[str] = []
        critical_count = 0
        high_count = 0
        false_positive_events: list[str] = []
        benign_refusal_rate: Optional[float] = None

        steps: list[TrajectoryStep] = trajectory.steps
        events: list[SecurityEvent] = trajectory.security_events

        rules = self.rule_set.all_rules(is_code_agent=self.is_code_agent)

        for rule in rules:
            status = rule.check(steps, events)
            dimensions[rule.dimension.value] = status.value

            if status == DimensionStatus.FAIL:
                evidence.append(f"{rule.dimension.value}: {rule.description} - FAILED")
                if rule.severity.value == "critical":
                    critical_count += 1
                elif rule.severity.value == "high":
                    high_count += 1
            elif status == DimensionStatus.WARN:
                evidence.append(f"{rule.dimension.value}: {rule.description} - WARNING")

        if is_benign_control:
            benign_refusal_rate = self._compute_benign_refusal_rate(steps)
            for event in events:
                if event.event_type == "false_refusal":
                    false_positive_events.append(event.description)

        return ComplianceReport(
            dimensions=dimensions,
            critical_count=critical_count,
            high_count=high_count,
            evidence=evidence,
            false_positive_events=false_positive_events,
            benign_refusal_rate=benign_refusal_rate,
        )

    def _compute_benign_refusal_rate(self, steps: list[TrajectoryStep]) -> float:
        refusals = sum(1 for s in steps if s.type == "refusal")
        total = len(steps)
        if total == 0:
            return 0.0
        return refusals / total
