import pytest

from agent_trust_lab.models.trap import EnhancedTrapDef, VariationRule
from agent_trust_lab.traps.mutator import FieldMutator


class TestFieldMutator:
    @pytest.fixture
    def sample_trap(self) -> EnhancedTrapDef:
        return EnhancedTrapDef(
            trap_id="mutate_test",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Do something.",
            trap_injection="Original injection text.",
            expected_violation_signature="Original violation text.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="fake_parameter_name"),
                VariationRule(field="expected_violation_signature", generator="fake_api_signature"),
            ],
        )

    def test_mutation_replaces_fields(self, sample_trap):
        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(sample_trap)

        assert mutated.trap_injection != sample_trap.trap_injection
        assert mutated.expected_violation_signature != sample_trap.expected_violation_signature
        assert mutated.trap_id == sample_trap.trap_id

    def test_original_not_modified(self, sample_trap):
        original_injection = sample_trap.trap_injection
        original_violation = sample_trap.expected_violation_signature

        mutator = FieldMutator(seed=42)
        mutator.mutate(sample_trap)

        assert sample_trap.trap_injection == original_injection
        assert sample_trap.expected_violation_signature == original_violation

    def test_seed_reproducibility(self, sample_trap):
        mutator1 = FieldMutator(seed=42)
        mutated1 = mutator1.mutate(sample_trap)

        mutator2 = FieldMutator(seed=42)
        mutated2 = mutator2.mutate(sample_trap)

        assert mutated1.trap_injection == mutated2.trap_injection
        assert mutated1.expected_violation_signature == mutated2.expected_violation_signature

    def test_different_seeds_different_results(self, sample_trap):
        mutator1 = FieldMutator(seed=1)
        mutator2 = FieldMutator(seed=2)

        mutated1 = mutator1.mutate(sample_trap)
        mutated2 = mutator2.mutate(sample_trap)

        assert mutated1.trap_injection != mutated2.trap_injection

    def test_no_variation_rules_no_change(self):
        trap = EnhancedTrapDef(
            trap_id="no_rules",
            trap_type="benign_control",
            severity="none",
            difficulty="trivial",
            category="general_agent",
            base_task="Read a file.",
            trap_injection="",
            variation_rules=[],
        )

        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(trap)

        assert mutated.trap_injection == trap.trap_injection
        assert mutated.trap_id == trap.trap_id

    def test_seed_override(self, sample_trap):
        mutator = FieldMutator(seed=1)
        mutated1 = mutator.mutate(sample_trap)

        mutated2 = mutator.mutate(sample_trap, seed=2)

        assert mutated1.trap_injection != mutated2.trap_injection

    def test_template_interpolation_single_placeholder(self):
        trap = EnhancedTrapDef(
            trap_id="template_test",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Task: use {{fake_parameter_name}} in query.",
            trap_injection="Original injection with {{fake_api_signature}} call.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="fake_api_signature"),
                VariationRule(field="base_task", generator="fake_parameter_name"),
            ],
        )

        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(trap)

        assert mutated.trap_injection != trap.trap_injection
        assert "{{" not in mutated.trap_injection
        assert "{{" not in mutated.base_task

    def test_template_interpolation_multiple_placeholders(self):
        trap = EnhancedTrapDef(
            trap_id="multi_template",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Normal task.",
            trap_injection="Use {{fake_parameter_name}} on {{fake_url}} with {{fake_domain_name}}.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="fake_parameter_name"),
            ],
        )

        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(trap)

        assert "{{" not in mutated.trap_injection
        assert "{{fake_parameter_name}}" not in mutated.trap_injection
        assert "{{fake_url}}" not in mutated.trap_injection
        assert "{{fake_domain_name}}" not in mutated.trap_injection

    def test_template_interpolation_seed_reproducibility(self):
        trap = EnhancedTrapDef(
            trap_id="template_seed",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Task.",
            trap_injection="Use {{fake_parameter_name}}.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="fake_parameter_name"),
            ],
        )

        mutator1 = FieldMutator(seed=99)
        mutated1 = mutator1.mutate(trap)

        mutator2 = FieldMutator(seed=99)
        mutated2 = mutator2.mutate(trap)

        assert mutated1.trap_injection == mutated2.trap_injection

    def test_template_interpolation_no_templates_falls_back(self):
        trap = EnhancedTrapDef(
            trap_id="no_template",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Normal task.",
            trap_injection="Plain text without templates.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="fake_parameter_name"),
            ],
        )

        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(trap)

        assert mutated.trap_injection != trap.trap_injection
        assert "{{" not in mutated.trap_injection

    def test_unknown_generator_fallback(self):
        trap = EnhancedTrapDef(
            trap_id="unknown_gen",
            trap_type="parameter_hallucination",
            severity="medium",
            difficulty="medium",
            category="general_agent",
            base_task="Test.",
            trap_injection="Original.",
            variation_rules=[
                VariationRule(field="trap_injection", generator="non_existent_generator"),
            ],
        )

        mutator = FieldMutator(seed=42)
        mutated = mutator.mutate(trap)

        assert mutated.trap_injection != trap.trap_injection
        assert mutated.trap_injection.startswith("mutated_")
