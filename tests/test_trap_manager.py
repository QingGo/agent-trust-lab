from agent_trust_lab.traps.manager import TrapManager


class TestTrapManager:
    def test_load_all_traps(self, trap_manager):
        assert trap_manager.trap_count == 3

    def test_filter_by_category(self, trap_manager):
        general = trap_manager.load_traps(category="general_agent")
        assert len(general) == 1
        assert general[0].trap_id == "test_trap_01"

        code = trap_manager.load_traps(category="code_agent")
        assert len(code) == 1
        assert code[0].trap_id == "test_trap_03"

    def test_filter_by_difficulty(self, trap_manager):
        hard = trap_manager.load_traps(difficulty="hard")
        assert len(hard) == 1
        assert hard[0].trap_id == "test_trap_03"

    def test_filter_by_trap_ids(self, trap_manager):
        traps = trap_manager.load_traps(trap_ids=["test_trap_01", "test_trap_99"])
        assert len(traps) == 1
        assert traps[0].trap_id == "test_trap_01"

    def test_exclude_controls_by_default(self, trap_manager):
        traps = trap_manager.load_traps()
        trap_ids = [t.trap_id for t in traps]
        assert "test_trap_02" not in trap_ids  # benign_control

    def test_include_controls(self, trap_manager):
        all_traps = trap_manager.load_traps(include_controls=True)
        assert len(all_traps) == 3

    def test_get_trap(self, trap_manager):
        trap = trap_manager.get_trap("test_trap_01")
        assert trap is not None
        assert trap.trap_type == "parameter_hallucination"

    def test_get_trap_not_found(self, trap_manager):
        trap = trap_manager.get_trap("nonexistent")
        assert trap is None

    def test_list_categories(self, trap_manager):
        cats = trap_manager.list_categories()
        assert "general_agent" in cats
        assert "code_agent" in cats

    def test_list_difficulties(self, trap_manager):
        diffs = trap_manager.list_difficulties()
        assert "medium" in diffs
        assert "hard" in diffs

    def test_list_trap_types(self, trap_manager):
        types = trap_manager.list_trap_types()
        assert "parameter_hallucination" in types
        assert "code_semantic_hallucination" in types

    def test_empty_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TrapManager(tmpdir)
            assert mgr.trap_count == 0

    def test_trap_ids_takes_priority(self, trap_manager):
        traps = trap_manager.load_traps(
            trap_ids=["test_trap_01"],
            category="code_agent",
        )
        assert len(traps) == 1
        assert traps[0].trap_id == "test_trap_01"

    def test_category_all(self, trap_manager):
        traps = trap_manager.load_traps(category="all", include_controls=True)
        assert len(traps) == 3


class TestRealTrapLibrary:
    """Tests against the actual trap library."""

    def test_all_traps_load(self, real_trap_manager):
        assert real_trap_manager.trap_count > 0

    def test_all_traps_have_ids(self, real_trap_manager):
        all_traps = real_trap_manager.load_traps(include_controls=True)
        for trap in all_traps:
            assert trap.trap_id, "Trap has no trap_id"
            assert trap.trap_type, f"Trap {trap.trap_id} has no trap_type"
            assert trap.base_task, f"Trap {trap.trap_id} has no base_task"

    def test_general_and_code_categories(self, real_trap_manager):
        general = real_trap_manager.load_traps(category="general_agent")
        code = real_trap_manager.load_traps(category="code_agent")
        assert len(general) > 0
        assert len(code) > 0

    def test_diffulty_filtering(self, real_trap_manager):
        hard = real_trap_manager.load_traps(difficulty="hard")
        trivial = real_trap_manager.load_traps(difficulty="trivial", include_controls=True)

        assert len(hard) > 0
        assert len(trivial) > 0
        for t in hard:
            assert t.difficulty == "hard"
        for t in trivial:
            assert t.difficulty == "trivial"

    def test_controls_excluded_by_default(self, real_trap_manager):
        without_controls = real_trap_manager.load_traps(include_controls=False)
        with_controls = real_trap_manager.load_traps(include_controls=True)
        assert len(with_controls) > len(without_controls)

    def test_control_types(self, real_trap_manager):
        all_traps = real_trap_manager.load_traps(include_controls=True)
        control_types = {"benign_control", "overly_cautious", "benign_code_control"}
        controls = [t for t in all_traps if t.trap_type in control_types]
        assert len(controls) > 0

    def test_total_trap_count(self, real_trap_manager):
        all_traps = real_trap_manager.load_traps(include_controls=True)
        assert len(all_traps) >= 70  # streamlined from 160 to 76, currently 71

    def test_attack_trap_count(self, real_trap_manager):
        attack_traps = real_trap_manager.load_traps(include_controls=False)
        assert len(attack_traps) >= 65

    def test_new_prompt_extraction_category(self, real_trap_manager):
        traps = real_trap_manager.load_traps(include_controls=True)
        types = [t.trap_type for t in traps if t.trap_type == "prompt_extraction"]
        assert len(types) >= 2

    def test_new_phishing_category(self, real_trap_manager):
        traps = real_trap_manager.load_traps(include_controls=True)
        types = [t.trap_type for t in traps if t.trap_type == "phishing_injection"]
        assert len(types) >= 1

    def test_new_dos_category(self, real_trap_manager):
        traps = real_trap_manager.load_traps(include_controls=True)
        types = [t.trap_type for t in traps if t.trap_type == "dos_attack"]
        assert len(types) >= 1

    def test_new_mcp_categories(self, real_trap_manager):
        traps = real_trap_manager.load_traps(include_controls=True)
        mcp_types = {t.trap_type for t in traps if t.trap_type.startswith("mcp_")}
        expected = {
            "mcp_tool_poisoning",
            "mcp_resource_exfiltration",
            "mcp_prompt_injection",
            "mcp_tool_impersonation",
        }
        assert mcp_types == expected

    def test_new_backdoor_category(self, real_trap_manager):
        traps = real_trap_manager.load_traps(include_controls=True)
        types = [t.trap_type for t in traps if t.trap_type == "backdoor_injection"]
        assert len(types) >= 2

    def test_general_trap_count(self, real_trap_manager):
        general = real_trap_manager.load_traps(category="general_agent")
        assert len(general) >= 50  # streamlined from 117 to 56

    def test_code_trap_count(self, real_trap_manager):
        code = real_trap_manager.load_traps(category="code_agent")
        assert len(code) >= 10  # streamlined from 16 to 12
