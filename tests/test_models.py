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

    def test_secure_trajectory_from_dict_roundtrip(self):
        original = SecureTrajectory(
            steps=[
                TrajectoryStep(type="thought", content="Thinking..."),
                TrajectoryStep(
                    type="action",
                    content="Running command",
                    tools_called=["shell"],
                    metadata={"tool_args": "ls"},
                ),
            ],
            security_events=[
                SecurityEvent(event_type="cmd_filtered", description="Blocked", step_index=1),
            ],
            dry_run_log="All good.",
            policy_rules_applied=["Unauthorized tool access"],
            actual_violations=["violation_1"],
            metadata={"adapter": "test", "model": "x", "stub": False},
        )

        data = original.to_dict()
        restored = SecureTrajectory.from_dict(data)

        assert len(restored.steps) == len(original.steps)
        assert len(restored.security_events) == len(original.security_events)
        assert restored.dry_run_log == original.dry_run_log
        assert restored.policy_rules_applied == original.policy_rules_applied
        assert restored.actual_violations == original.actual_violations
        assert restored.metadata == original.metadata

        assert restored.steps[0].type == "thought"
        assert restored.steps[0].content == "Thinking..."
        assert restored.steps[1].tools_called == ["shell"]
        assert restored.steps[1].metadata == {"tool_args": "ls"}

        assert restored.security_events[0].event_type == "cmd_filtered"

    def test_secure_trajectory_from_dict_empty(self):
        traj = SecureTrajectory.from_dict({"steps": []})
        assert traj.steps == []
        assert traj.security_events == []
        assert traj.dry_run_log == ""
        assert traj.policy_rules_applied == []
        assert traj.metadata == {}

    def test_secure_trajectory_from_dict_minimal(self):
        traj = SecureTrajectory.from_dict(
            {
                "steps": [{"type": "error", "content": "test error"}],
            }
        )
        assert len(traj.steps) == 1
        assert traj.steps[0].type == "error"
        assert traj.steps[0].content == "test error"


class TestReportModels:
    def test_hallu_step_report_anchor_type_default(self):
        from agent_trust_lab.models.report import HalluStepReport

        step = HalluStepReport(step_index=0, gsar_label="Grounded")
        assert step.anchor_type == "none"

    def test_hallu_step_report_anchor_type_custom(self):
        from agent_trust_lab.models.report import HalluStepReport

        step = HalluStepReport(step_index=0, gsar_label="Grounded", anchor_type="semantic")
        assert step.anchor_type == "semantic"

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

    def test_skip_extract_types_defaults(self):
        config = EvaluationConfig()
        assert config.skip_extract_types == ["action", "error"]

    def test_skip_extract_types_custom(self):
        config = EvaluationConfig(skip_extract_types=["action"])
        assert config.skip_extract_types == ["action"]

    def test_grounded_threshold_default(self):
        config = EvaluationConfig()
        assert config.grounded_threshold == 0.3

    def test_grounded_threshold_custom(self):
        config = EvaluationConfig(grounded_threshold=0.7)
        assert config.grounded_threshold == 0.7

    def test_grounded_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="grounded_threshold"):
            EvaluationConfig(grounded_threshold=-0.1)

    def test_grounded_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="grounded_threshold"):
            EvaluationConfig(grounded_threshold=1.5)

    def test_nli_neutral_weight_default(self):
        config = EvaluationConfig()
        assert config.nli_neutral_weight == 0.5

    def test_nli_neutral_weight_custom(self):
        config = EvaluationConfig(nli_neutral_weight=0.8)
        assert config.nli_neutral_weight == 0.8

    def test_nli_neutral_weight_below_zero_raises(self):
        with pytest.raises(ValueError, match="nli_neutral_weight"):
            EvaluationConfig(nli_neutral_weight=-0.1)

    def test_nli_neutral_weight_above_one_raises(self):
        with pytest.raises(ValueError, match="nli_neutral_weight"):
            EvaluationConfig(nli_neutral_weight=1.5)

    def test_anchor_type_weights_default(self):
        config = EvaluationConfig()
        assert config.anchor_type_weights == {
            "semantic": 0.7,
            "token_overlap": 0.6,
            "multi_hop": 0.6,
            "none": 0.5,
        }

    def test_anchor_type_weights_custom(self):
        config = EvaluationConfig(
            anchor_type_weights={
                "semantic": 0.8,
                "token_overlap": 0.4,
                "multi_hop": 0.5,
                "none": 0.3,
            }
        )
        assert config.anchor_type_weights["semantic"] == 0.8

    def test_anchor_type_weights_below_zero_raises(self):
        with pytest.raises(ValueError, match="anchor_type_weights"):
            EvaluationConfig(anchor_type_weights={"none": -0.1})

    def test_anchor_type_weights_above_one_raises(self):
        with pytest.raises(ValueError, match="anchor_type_weights"):
            EvaluationConfig(anchor_type_weights={"semantic": 1.5})

    def test_max_steps_below_one_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            EvaluationConfig(max_steps=0)

    def test_parallel_below_one_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            EvaluationConfig(parallel=0)

    def test_timeout_below_one_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            EvaluationConfig(timeout=0)

    def test_judge_model_defaults_flash(self):
        config = EvaluationConfig()
        assert config.judge_model == "deepseek-v4-flash"

    def test_judge_model_custom(self):
        config = EvaluationConfig(judge_model="deepseek-v4-pro")
        assert config.judge_model == "deepseek-v4-pro"

    def test_cache_enabled_default(self):
        config = EvaluationConfig()
        assert config.cache_enabled is True

    def test_cache_ttl_days_default(self):
        config = EvaluationConfig()
        assert config.cache_ttl_days == 7

    def test_cache_dir_default(self):
        from agent_trust_lab.config import RESULT_CACHE_DIR

        config = EvaluationConfig()
        assert config.cache_dir == RESULT_CACHE_DIR

    def test_cache_dir_custom(self):
        config = EvaluationConfig(cache_dir="/tmp/my_cache")
        assert config.cache_dir == "/tmp/my_cache"

    def test_cache_ttl_days_below_zero_raises(self):
        with pytest.raises(ValueError, match="cache_ttl_days"):
            EvaluationConfig(cache_ttl_days=-1)

    def test_adaptive_sampling_default(self):
        config = EvaluationConfig()
        assert config.adaptive_sampling is True

    def test_adaptive_disagreement_threshold_default(self):
        config = EvaluationConfig()
        assert config.adaptive_disagreement_threshold == 0.3

    def test_adaptive_disagreement_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="adaptive_disagreement_threshold"):
            EvaluationConfig(adaptive_disagreement_threshold=-0.1)

    def test_adaptive_disagreement_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="adaptive_disagreement_threshold"):
            EvaluationConfig(adaptive_disagreement_threshold=1.5)

    def test_adaptive_max_samples_default(self):
        config = EvaluationConfig()
        assert config.adaptive_max_samples == 3

    def test_adaptive_max_samples_below_one_raises(self):
        with pytest.raises(ValueError, match="adaptive_max_samples"):
            EvaluationConfig(adaptive_max_samples=0)

    def test_self_consistency_enabled_default(self):
        config = EvaluationConfig()
        assert config.self_consistency_enabled is False

    def test_self_consistency_samples_default(self):
        config = EvaluationConfig()
        assert config.self_consistency_samples == 5

    def test_self_consistency_samples_below_two_raises(self):
        with pytest.raises(ValueError, match="self_consistency_samples"):
            EvaluationConfig(self_consistency_enabled=True, self_consistency_samples=1)

    def test_temperature_default(self):
        config = EvaluationConfig()
        assert config.temperature == 0.0

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            EvaluationConfig(temperature=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            EvaluationConfig(temperature=2.1)

    def test_temperature_custom(self):
        config = EvaluationConfig(temperature=0.7)
        assert config.temperature == 0.7

    def test_difficulty_weights_default(self):
        config = EvaluationConfig()
        assert config.difficulty_weights == {
            "trivial": 0.25, "easy": 0.5, "medium": 0.75, "hard": 1.0
        }

    def test_difficulty_weights_custom(self):
        config = EvaluationConfig(difficulty_weights={"hard": 1.0, "medium": 0.5})
        assert config.difficulty_weights["hard"] == 1.0
        assert config.difficulty_weights["medium"] == 0.5

    def test_difficulty_weights_below_zero_raises(self):
        with pytest.raises(ValueError, match="difficulty_weights"):
            EvaluationConfig(difficulty_weights={"hard": -0.1})

    def test_difficulty_weights_above_one_raises(self):
        with pytest.raises(ValueError, match="difficulty_weights"):
            EvaluationConfig(difficulty_weights={"hard": 1.1})

    def test_injection_template_default(self):
        config = EvaluationConfig()
        assert config.injection_template == "system_note"

    def test_injection_template_custom(self):
        config = EvaluationConfig(injection_template="colleague")
        assert config.injection_template == "colleague"

    def test_with_base_url_auto_judge_model(self):
        config = EvaluationConfig(model="mimo-v2.5-pro", base_url="https://example.com/v1")
        assert config.judge_model == "deepseek-v4-flash"

    def test_with_base_url_explicit_judge_model(self):
        config = EvaluationConfig(
            model="mimo-v2.5-pro",
            base_url="https://example.com/v1",
            judge_model="gpt-4o",
        )
        assert config.judge_model == "gpt-4o"
