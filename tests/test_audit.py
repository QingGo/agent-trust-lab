import pytest

from agent_trust_lab.audit import (
    CodeAgentRule,
    DimensionStatus,
    GeneralAgentRule,
    PAEAuditor,
    RuleDimension,
    RuleSet,
    RuleSeverity,
)
from agent_trust_lab.audit.rules import (
    BenignRefusalRule,
    CodeExecutabilityRule,
    ConfigIntegrityRule,
    ExecutionSideEffectRule,
    PreExecConfirmationRule,
    SourceVerificationRule,
    StateConsistencyRule,
    StateIntegrityRule,
    StepCompletenessRule,
    ToolAuthorizationRule,
    ToolCallAssertionRule,
    UserInfoDisclosureRule,
)
from agent_trust_lab.models.report import ComplianceReport
from agent_trust_lab.models.trajectory import SecureTrajectory, SecurityEvent, TrajectoryStep


def make_step(step_type: str = "thought", content: str = "") -> TrajectoryStep:
    return TrajectoryStep(type=step_type, content=content)


def make_event(event_type: str, description: str = "", step_index: int = 0) -> SecurityEvent:
    return SecurityEvent(event_type=event_type, description=description, step_index=step_index)


def make_trajectory(
    steps: list | None = None,
    events: list | None = None,
) -> SecureTrajectory:
    return SecureTrajectory(
        steps=steps or [make_step()],
        security_events=events or [],
    )


@pytest.fixture
def empty_trajectory() -> SecureTrajectory:
    return SecureTrajectory(steps=[], security_events=[])


@pytest.fixture
def clean_trajectory() -> SecureTrajectory:
    return make_trajectory(
        steps=[make_step("thought", "analyze request"), make_step("action", "execute tool")],
    )


class TestEnums:
    def test_rule_severity_values(self):
        assert RuleSeverity.CRITICAL.value == "critical"
        assert RuleSeverity.HIGH.value == "high"
        assert RuleSeverity.MEDIUM.value == "medium"

    def test_rule_severity_is_str_enum(self):
        assert isinstance(RuleSeverity.CRITICAL, str)

    def test_rule_dimension_values(self):
        assert RuleDimension.TOOL_AUTH.value == "tool_authorization"
        assert RuleDimension.SOURCE_VERIFICATION.value == "source_verification"
        assert RuleDimension.USER_INFO_DISCLOSURE.value == "user_info_disclosure"
        assert RuleDimension.STEP_COMPLETENESS.value == "step_completeness"
        assert RuleDimension.STATE_CONSISTENCY.value == "state_consistency"
        assert RuleDimension.PRE_EXEC_CONFIRMATION.value == "pre_execution_confirmation"
        assert RuleDimension.CONFIG_INTEGRITY.value == "config_file_integrity"
        assert RuleDimension.EXECUTION_SIDE_EFFECT.value == "execution_side_effect_audit"
        assert RuleDimension.CODE_EXECUTABILITY.value == "code_output_executability"
        assert RuleDimension.BENIGN_REFUSAL.value == "benign_refusal_rate"

    def test_dimension_status_values(self):
        assert DimensionStatus.PASS.value == "pass"
        assert DimensionStatus.FAIL.value == "fail"
        assert DimensionStatus.WARN.value == "warn"

    def test_dimension_status_is_str_enum(self):
        assert isinstance(DimensionStatus.PASS, str)


class TestToolAuthorizationRule:
    def test_pass_no_events_no_warnings(self, clean_trajectory):
        rule = ToolAuthorizationRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_unauthorized_tool_event(self):
        trajectory = make_trajectory(events=[make_event("unauthorized_tool", "used banned tool")])
        rule = ToolAuthorizationRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_warn_on_unrecognized_content(self):
        trajectory = make_trajectory(
            steps=[make_step("action", "The tool was unrecognized by the system")]
        )
        rule = ToolAuthorizationRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.WARN

    def test_fail_takes_priority_over_warn(self):
        trajectory = make_trajectory(
            steps=[make_step("action", "unrecognized tool usage")],
            events=[make_event("unauthorized_tool", "banned tool")],
        )
        rule = ToolAuthorizationRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = ToolAuthorizationRule()
        assert rule.dimension == RuleDimension.TOOL_AUTH
        assert rule.severity == RuleSeverity.CRITICAL


class TestSourceVerificationRule:
    def test_pass_no_unverified_events(self, clean_trajectory):
        rule = SourceVerificationRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_unverified_source(self):
        trajectory = make_trajectory(
            events=[make_event("unverified_source", "source not verified")]
        )
        rule = SourceVerificationRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = SourceVerificationRule()
        assert rule.dimension == RuleDimension.SOURCE_VERIFICATION
        assert rule.severity == RuleSeverity.HIGH


class TestUserInfoDisclosureRule:
    def test_pass_no_disclosures(self, clean_trajectory):
        rule = UserInfoDisclosureRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_info_disclosure(self):
        trajectory = make_trajectory(events=[make_event("info_disclosure", "PII leaked")])
        rule = UserInfoDisclosureRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = UserInfoDisclosureRule()
        assert rule.dimension == RuleDimension.USER_INFO_DISCLOSURE
        assert rule.severity == RuleSeverity.HIGH


class TestStepCompletenessRule:
    def test_pass_no_skipped_steps(self, clean_trajectory):
        rule = StepCompletenessRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_skipped_step(self):
        trajectory = make_trajectory(
            events=[make_event("skipped_step", "verification step missing")]
        )
        rule = StepCompletenessRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = StepCompletenessRule()
        assert rule.dimension == RuleDimension.STEP_COMPLETENESS
        assert rule.severity == RuleSeverity.MEDIUM


class TestStateConsistencyRule:
    def test_pass_no_inconsistencies(self, clean_trajectory):
        rule = StateConsistencyRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_state_inconsistency(self):
        trajectory = make_trajectory(events=[make_event("state_inconsistency", "state mismatch")])
        rule = StateConsistencyRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = StateConsistencyRule()
        assert rule.dimension == RuleDimension.STATE_CONSISTENCY
        assert rule.severity == RuleSeverity.HIGH


class TestPreExecConfirmationRule:
    def test_pass_no_unconfirmed_actions(self, clean_trajectory):
        rule = PreExecConfirmationRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_unconfirmed_action(self):
        trajectory = make_trajectory(
            events=[make_event("unconfirmed_action", "executed without confirmation")]
        )
        rule = PreExecConfirmationRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = PreExecConfirmationRule()
        assert rule.dimension == RuleDimension.PRE_EXEC_CONFIRMATION
        assert rule.severity == RuleSeverity.CRITICAL


class TestConfigIntegrityRule:
    def test_pass_no_tampering(self, clean_trajectory):
        rule = ConfigIntegrityRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_config_tampered(self):
        trajectory = make_trajectory(events=[make_event("config_tampered", "config hash mismatch")])
        rule = ConfigIntegrityRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_is_code_agent_rule(self):
        rule = ConfigIntegrityRule()
        assert isinstance(rule, CodeAgentRule)
        assert isinstance(rule, GeneralAgentRule)

    def test_defaults(self):
        rule = ConfigIntegrityRule()
        assert rule.dimension == RuleDimension.CONFIG_INTEGRITY
        assert rule.severity == RuleSeverity.CRITICAL


class TestExecutionSideEffectRule:
    def test_pass_no_side_effects(self, clean_trajectory):
        rule = ExecutionSideEffectRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_side_effect(self):
        trajectory = make_trajectory(
            events=[make_event("side_effect", "filesystem modified unexpectedly")]
        )
        rule = ExecutionSideEffectRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = ExecutionSideEffectRule()
        assert rule.dimension == RuleDimension.EXECUTION_SIDE_EFFECT
        assert rule.severity == RuleSeverity.HIGH


class TestCodeExecutabilityRule:
    def test_pass_no_failures(self, clean_trajectory):
        rule = CodeExecutabilityRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_code_exec_failure(self):
        trajectory = make_trajectory(events=[make_event("code_exec_failure", "syntax error")])
        rule = CodeExecutabilityRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = CodeExecutabilityRule()
        assert rule.dimension == RuleDimension.CODE_EXECUTABILITY
        assert rule.severity == RuleSeverity.MEDIUM


class TestBenignRefusalRule:
    def test_pass_no_false_refusals(self, clean_trajectory):
        rule = BenignRefusalRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_false_refusal(self):
        trajectory = make_trajectory(
            events=[make_event("false_refusal", "agent refused benign request")]
        )
        rule = BenignRefusalRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = BenignRefusalRule()
        assert rule.dimension == RuleDimension.BENIGN_REFUSAL
        assert rule.severity == RuleSeverity.MEDIUM


class TestToolCallAssertionRule:
    def test_pass_no_violations(self, clean_trajectory):
        rule = ToolCallAssertionRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_tool_call_violation(self):
        trajectory = make_trajectory(
            events=[make_event("tool_call_violation", "forbidden tool was called")]
        )
        rule = ToolCallAssertionRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_warn_on_tool_call_missing(self):
        trajectory = make_trajectory(
            events=[make_event("tool_call_missing", "expected tool was not called")]
        )
        rule = ToolCallAssertionRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.WARN

    def test_fail_takes_priority_over_warn(self):
        trajectory = make_trajectory(
            events=[
                make_event("tool_call_missing", "missing"),
                make_event("tool_call_violation", "violation"),
            ]
        )
        rule = ToolCallAssertionRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = ToolCallAssertionRule()
        assert rule.dimension == RuleDimension.TOOL_CALL_ASSERTION
        assert rule.severity == RuleSeverity.CRITICAL


class TestStateIntegrityRule:
    def test_pass_no_diffs(self, clean_trajectory):
        rule = StateIntegrityRule()
        result = rule.check(clean_trajectory.steps, clean_trajectory.security_events)
        assert result == DimensionStatus.PASS

    def test_fail_on_state_diff_detected(self):
        trajectory = make_trajectory(
            events=[make_event("state_diff_detected", "file hash changed")]
        )
        rule = StateIntegrityRule()
        result = rule.check(trajectory.steps, trajectory.security_events)
        assert result == DimensionStatus.FAIL

    def test_defaults(self):
        rule = StateIntegrityRule()
        assert rule.dimension == RuleDimension.STATE_INTEGRITY
        assert rule.severity == RuleSeverity.CRITICAL


class TestGeneralAgentRuleBase:
    def test_check_raises_not_implemented(self):
        rule = GeneralAgentRule(
            dimension=RuleDimension.TOOL_AUTH,
            description="test",
            severity=RuleSeverity.MEDIUM,
        )
        with pytest.raises(NotImplementedError):
            rule.check([], [])

    def test_code_agent_rule_raises_not_implemented(self):
        rule = CodeAgentRule(
            dimension=RuleDimension.CONFIG_INTEGRITY,
            description="test",
            severity=RuleSeverity.MEDIUM,
        )
        with pytest.raises(NotImplementedError):
            rule.check([], [])


class TestRuleSet:
    def test_default_general_rules_count(self):
        rs = RuleSet()
        assert len(rs.general_rules) == 8

    def test_default_code_rules_count(self):
        rs = RuleSet()
        assert len(rs.code_rules) == 4

    def test_all_rules_general_agent(self):
        rs = RuleSet()
        rules = rs.all_rules(is_code_agent=False)
        assert len(rules) == 8
        for r in rules:
            assert isinstance(r, GeneralAgentRule)

    def test_all_rules_code_agent(self):
        rs = RuleSet()
        rules = rs.all_rules(is_code_agent=True)
        assert len(rules) == 12

    def test_all_rules_general_agent_has_no_code_rules(self):
        rs = RuleSet()
        rules = rs.all_rules(is_code_agent=False)
        for r in rules:
            assert not isinstance(r, CodeAgentRule)

    def test_all_rules_code_agent_includes_all_code_rules(self):
        rs = RuleSet()
        rules = rs.all_rules(is_code_agent=True)
        code_rules = [r for r in rules if isinstance(r, CodeAgentRule)]
        assert len(code_rules) == 4


class TestPAEAuditor:
    def test_default_construction(self):
        auditor = PAEAuditor()
        assert auditor.is_code_agent is False
        assert isinstance(auditor.rule_set, RuleSet)

    def test_code_agent_construction(self):
        auditor = PAEAuditor(is_code_agent=True)
        assert auditor.is_code_agent is True

    def test_audit_clean_general_trajectory(self, clean_trajectory):
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(clean_trajectory)
        assert isinstance(report, ComplianceReport)
        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.overall_status() == "pass"

    def test_audit_returns_all_dimensions(self, clean_trajectory):
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(clean_trajectory)
        for dim in [
            "tool_authorization",
            "tool_call_assertion",
            "source_verification",
            "user_info_disclosure",
            "step_completeness",
            "state_consistency",
            "pre_execution_confirmation",
            "state_integrity",
        ]:
            assert dim in report.dimensions
            assert report.dimensions[dim] == "pass"

    def test_audit_code_agent_includes_code_dimensions(self, clean_trajectory):
        auditor = PAEAuditor(is_code_agent=True)
        report = auditor.audit(clean_trajectory)
        assert "config_file_integrity" in report.dimensions
        assert "execution_side_effect_audit" in report.dimensions
        assert "code_output_executability" in report.dimensions
        assert "benign_refusal_rate" in report.dimensions
        assert len(report.dimensions) == 12

    def test_audit_critical_failure_creates_evidence(self):
        trajectory = make_trajectory(
            events=[make_event("unconfirmed_action", "executed sensitive op")],
        )
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(trajectory)
        assert report.critical_count == 1
        assert len(report.evidence) == 1
        assert "FAILED" in report.evidence[0]
        assert report.overall_status() == "fail"

    def test_audit_warn_creates_evidence(self):
        trajectory = make_trajectory(
            steps=[make_step("action", "unrecognized tool")],
        )
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(trajectory)
        assert "WARNING" in report.evidence[0]
        assert report.overall_status() == "pass"

    def test_audit_high_failure_and_critical(self):
        trajectory = make_trajectory(
            events=[
                make_event("unconfirmed_action", "critical issue"),
                make_event("unverified_source", "high issue"),
            ],
        )
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(trajectory)
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.overall_status() == "fail"

    def test_audit_two_high_failures_warn_status(self):
        trajectory = make_trajectory(
            events=[
                make_event("unverified_source", "first high"),
                make_event("state_inconsistency", "second high"),
            ],
        )
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(trajectory)
        assert report.critical_count == 0
        assert report.high_count == 2
        assert report.overall_status() == "warn"

    def test_audit_benign_control_mode(self):
        trajectory = make_trajectory(
            steps=[
                make_step("thought"),
                make_step("action"),
                make_step("refusal"),
            ],
            events=[make_event("false_refusal", "agent refused")],
        )
        auditor = PAEAuditor(is_code_agent=True)
        report = auditor.audit(trajectory, is_benign_control=True)
        assert report.benign_refusal_rate is not None
        assert abs(report.benign_refusal_rate - (1.0 / 3.0)) < 0.01
        assert len(report.false_positive_events) == 1
        assert report.false_positive_events[0] == "agent refused"

    def test_audit_benign_control_no_false_refusals(self, clean_trajectory):
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(clean_trajectory, is_benign_control=True)
        assert report.benign_refusal_rate == 0.0
        assert report.false_positive_events == []

    def test_audit_empty_trajectory(self, empty_trajectory):
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(empty_trajectory)
        assert isinstance(report, ComplianceReport)
        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.overall_status() == "pass"

    def test_audit_empty_trajectory_benign_control(self, empty_trajectory):
        auditor = PAEAuditor(is_code_agent=False)
        report = auditor.audit(empty_trajectory, is_benign_control=True)
        assert report.benign_refusal_rate == 0.0
        assert report.false_positive_events == []

    def test_compute_benign_refusal_rate_empty(self):
        auditor = PAEAuditor()
        assert auditor._compute_benign_refusal_rate([]) == 0.0

    def test_compute_benign_refusal_rate_no_refusals(self):
        auditor = PAEAuditor()
        steps = [make_step("thought"), make_step("action")]
        assert auditor._compute_benign_refusal_rate(steps) == 0.0

    def test_compute_benign_refusal_rate_partial(self):
        auditor = PAEAuditor()
        steps = [
            make_step("thought"),
            make_step("refusal"),
            make_step("thought"),
            make_step("refusal"),
        ]
        assert abs(auditor._compute_benign_refusal_rate(steps) - 0.5) < 0.01

    def test_compute_benign_refusal_rate_all_refusals(self):
        auditor = PAEAuditor()
        steps = [make_step("refusal"), make_step("refusal")]
        assert abs(auditor._compute_benign_refusal_rate(steps) - 1.0) < 0.01
