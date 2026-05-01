import os
from typing import List, Optional

import yaml

from agent_trust_lab.models.trap import EnhancedTrapDef


class TrapManager:
    """Loads and manages traps from a YAML trap library."""

    def __init__(self, trap_library_path: str):
        self.trap_library_path = trap_library_path
        self._traps: dict[str, EnhancedTrapDef] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Recursively load all YAML traps from the library path."""
        if not os.path.isdir(self.trap_library_path):
            return
        for root, _dirs, files in os.walk(self.trap_library_path):
            for filename in files:
                if filename.endswith((".yaml", ".yml")):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                        if data is None:
                            continue
                        trap = EnhancedTrapDef(**data)
                        self._traps[trap.trap_id] = trap
                    except Exception as e:
                        print(f"Warning: Failed to load trap from {filepath}: {e}")

    def load_traps(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        include_controls: bool = False,
    ) -> List[EnhancedTrapDef]:
        """Load traps filtered by criteria.

        Args:
            trap_ids: Specific trap IDs to load. Overrides other filters if set.
            category: Filter by category ('general_agent', 'code_agent', or None for all).
            difficulty: Filter by difficulty ('trivial', 'easy', 'medium', 'hard').
            include_controls: Include benign control and overly cautious samples.
        """
        if trap_ids is not None:
            return [t for tid in trap_ids if (t := self._traps.get(tid)) is not None]

        results = list(self._traps.values())

        if category is not None and category != "all":
            results = [t for t in results if t.category == category]

        if difficulty is not None:
            results = [t for t in results if t.difficulty == difficulty]

        if not include_controls:
            excluded_types = {"benign_control", "overly_cautious", "benign_code_control"}
            results = [t for t in results if t.trap_type not in excluded_types]

        return results

    def get_trap(self, trap_id: str) -> Optional[EnhancedTrapDef]:
        """Get a single trap by ID."""
        return self._traps.get(trap_id)

    def list_categories(self) -> List[str]:
        """Return all distinct categories in the trap library."""
        return sorted(set(t.category for t in self._traps.values()))

    def list_difficulties(self) -> List[str]:
        """Return all distinct difficulty levels in the trap library."""
        return sorted(set(t.difficulty for t in self._traps.values()))

    def list_trap_types(self) -> List[str]:
        """Return all distinct trap types."""
        return sorted(set(t.trap_type for t in self._traps.values()))

    @property
    def trap_count(self) -> int:
        return len(self._traps)

    @staticmethod
    def _load_single_file(filepath: str) -> Optional[EnhancedTrapDef]:
        """Load a single trap from a YAML file path. Returns None on failure."""
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                return None
            return EnhancedTrapDef(**data)
        except Exception as e:
            print(f"Warning: Failed to load trap from {filepath}: {e}")
            return None

    def apply_mutation(
        self,
        trap: EnhancedTrapDef,
        seed: Optional[int] = None,
    ) -> EnhancedTrapDef:
        """Apply field-level mutation to a trap, returning a new instance."""
        from agent_trust_lab.traps.mutator import FieldMutator

        mutator = FieldMutator(seed=seed)
        return mutator.mutate(trap)
