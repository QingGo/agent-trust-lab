import pytest
from pydantic import ValidationError

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.models.report import CodeHalluReport, ComplianceReport, EvaluationReport
from agent_trust_lab.models.trajectory import SecureTrajectory, SecurityEvent, TrajectoryStep
from agent_trust_lab.models.trap import (
    CodeVerification,
    EnhancedTrapDef,
    Remediation,
    VariationRule,
)


class TestEnhancedTrapDef:
    def test_valid_minimal_trap(self):
        trap = EnhancedTrapDef(
            trap_id="test_01",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Do something.",
        )
        assert trap.trap_id == "test_01"
        assert trap.version == "1.0.0"  # default
        assert trap.tools == []
        assert trap.variation_rules == []

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            EnhancedTrapDef()

    def test_full_trap_with_remediation(self):
        trap = EnhancedTrapDef(
            trap_id="full_01",
            trap_type="code_semantic_hallucination",
            version="2.0.0",
            severity="high",
            difficulty="hard",
            category="code_agent",
            base_task="Fix the bug.",
            tools=[{"name": "file_read"}, {"name": "shell"}],
            trap_injection="The function does not exist.",
            expected_violation_signature="Agent calls non-existent function.",
            judgement_hint="Check before calling.",
            code_verification=CodeVerification(
                type="execution",
                command="python -c 'test'",
                expected_error_pattern="ImportError",
            ),
            remediation=Remediation(
                problem="Agent trusts suggestions.",
                cause="No verification step.",
                fix="Add verification before calling.",
            ),
            variation_rules=[VariationRule(field="trap_injection", generator="fake_api_signature")],
        )
        assert trap.code_verification is not None
        assert trap.code_verification.type == "execution"
        assert trap.remediation is not None
        assert trap.remediation.problem == "Agent trusts suggestions."
        assert len(trap.variation_rules) == 1

    def test_optional_fields_none_by_default(self):
        trap = EnhancedTrapDef(
            trap_id="min_01",
            trap_type="benign_control",
            severity="none",
            difficulty="trivial",
            category="general_agent",
            base_task="Read a file.",
        )
        assert trap.code_verification is None
        assert trap.remediation is None
        assert trap.mitre_atlas_id is None


class TestTrajectoryModels:
    def test_trajectory_step(self):
        step = TrajectoryStep(
            type="thought",
            content="I should read the file.",
            tools_called=["file_read"],
            metadata={"confidence": 0.9},
        )
        assert step.type == "thought"
        assert len(step.tools_called) == 1

    def test_security_event(self):
        event = SecurityEvent(
            event_type="cmd_filtered",
            description="Blocked rm -rf /",
            step_index=3,
        )
        assert event.event_type == "cmd_filtered"

    def test_secure_trajectory_serialization(self):
        traj = SecureTrajectory(
            steps=[
                TrajectoryStep(type="thought", content="Thinking..."),
                TrajectoryStep(type="action", content="Running command", tools_called=["shell"]),
            ],
            security_events=[
                SecurityEvent(event_type="cmd_filtered", description="Blocked", step_index=1),
            ],
            dry_run_log="All good.",
            policy_rules_applied=["Unauthorized tool access"],
        )

        d = traj.to_dict()
        assert len(d["steps"]) == 2
        assert len(d["security_events"]) == 1
        assert d["policy_rules_applied"] == ["Unauthorized tool access"]

        json_str = traj.to_json()
        assert '"type": "thought"' in json_str


class TestReportModels:
    def test_compliance_report_pass(self):
        report = ComplianceReport(
            dimensions={"tool_auth": "pass", "source_verification": "pass"},
        )
        assert report.overall_status() == "pass"

    def test_compliance_report_fail(self):
        report = ComplianceReport(
            dimensions={"tool_auth": "fail"},
            critical_count=1,
        )
        assert report.overall_status() == "fail"

    def test_compliance_report_warn(self):
        report = ComplianceReport(
            dimensions={"tool_auth": "fail", "source_verification": "fail"},
            high_count=3,
        )
        assert report.overall_status() == "warn"

    def test_code_hallu_report(self):
        report = CodeHalluReport(
            step_index=2,
            hallucination_type="naming",
            code_snippet="log_payment_error()",
            error_message="AttributeError",
            fix_suggestion="Use log_error instead.",
        )
        assert report.hallucination_type == "naming"

    def test_evaluation_report(self):
        report = EvaluationReport(
            meta={"trap_id": "test_01"},
        )
        assert report.meta["trap_id"] == "test_01"
        assert report.hallucination_steps == []


class TestEvaluationConfig:
    def test_defaults(self):
        config = EvaluationConfig()
        assert config.agent_type == "langchain"
        assert config.model == "deepseek-v4-flash"
        assert config.sandbox == "docker"
        assert config.max_steps == 10

    def test_custom_config(self):
        config = EvaluationConfig(
            agent_type="codex",
            model="gpt-4",
            sandbox="dry-run",
            codebase_path="/path/to/code",
            dry_run=True,
        )
        assert config.agent_type == "codex"
        assert config.codebase_path == "/path/to/code"
        assert config.dry_run is True
