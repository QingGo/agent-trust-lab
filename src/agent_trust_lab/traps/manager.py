import os
from typing import List, Optional

import pydantic
import yaml

from agent_trust_lab.log import get_logger
from agent_trust_lab.models.trap import EnhancedTrapDef

logger = get_logger("traps.manager")


class TrapManager:
    """Loads and manages traps from a YAML trap library."""

    def __init__(self, trap_library_path: str):
        self.trap_library_path = trap_library_path
        self._traps: dict[str, EnhancedTrapDef] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Recursively load all YAML traps from the library path."""
        if not os.path.isdir(self.trap_library_path):
            logger.warning("Trap library path does not exist: %s", self.trap_library_path)
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
                    except (yaml.YAMLError, pydantic.ValidationError) as e:
                        logger.warning("Failed to load trap from %s: %s", filepath, e)
                    except Exception as e:
                        logger.warning("Failed to load trap from %s: %s", filepath, e)

    def load_traps(
        self,
        trap_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        trap_types: Optional[List[str]] = None,
        include_controls: bool = False,
    ) -> List[EnhancedTrapDef]:
        if trap_ids is not None:
            return [t for tid in trap_ids if (t := self._traps.get(tid)) is not None]

        results = list(self._traps.values())

        if category is not None and category != "all":
            results = [t for t in results if t.category == category]

        if difficulty is not None:
            results = [t for t in results if t.difficulty == difficulty]

        if trap_types is not None:
            results = [t for t in results if t.trap_type in trap_types]

        if not include_controls:
            excluded_types = {"benign_control", "overly_cautious", "benign_code_control"}
            results = [t for t in results if t.trap_type not in excluded_types]

        return results

    def get_trap(self, trap_id: str) -> Optional[EnhancedTrapDef]:
        return self._traps.get(trap_id)

    def list_categories(self) -> List[str]:
        return sorted(set(t.category for t in self._traps.values()))

    def list_difficulties(self) -> List[str]:
        return sorted(set(t.difficulty for t in self._traps.values()))

    def list_trap_types(self) -> List[str]:
        return sorted(set(t.trap_type for t in self._traps.values()))

    @property
    def trap_count(self) -> int:
        return len(self._traps)

    @staticmethod
    def _load_single_file(filepath: str) -> Optional[EnhancedTrapDef]:
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                return None
            return EnhancedTrapDef(**data)
        except (yaml.YAMLError, pydantic.ValidationError) as e:
            logger.warning("Failed to load trap from %s: %s", filepath, e)
            return None

    def apply_mutation(
        self,
        trap: EnhancedTrapDef,
        seed: Optional[int] = None,
    ) -> EnhancedTrapDef:
        from agent_trust_lab.traps.mutator import FieldMutator
        from agent_trust_lab.traps.structural_mutator import StructuralMutator

        trap = StructuralMutator(seed=seed).mutate(trap)
        mutator = FieldMutator(seed=seed)
        return mutator.mutate(trap)
