"""Compliance auditing for orchestrator pipeline.

Wraps PAEAuditor and signature matching logic.
"""

from agent_trust_lab.models.report import ComplianceReport
from agent_trust_lab.models.trajectory import SecureTrajectory


class ComplianceAuditor:
    """Audits agent trajectory against compliance rules."""

    def audit(
        self,
        trajectory: SecureTrajectory,
        is_code_agent: bool = False,
        is_benign: bool = False,
    ) -> ComplianceReport:
        from agent_trust_lab.audit.auditor import PAEAuditor

        auditor = PAEAuditor(is_code_agent=is_code_agent)
        return auditor.audit(trajectory, is_benign_control=is_benign)
