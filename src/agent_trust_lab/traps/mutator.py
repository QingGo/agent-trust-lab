import copy
import random
import re
import uuid
from typing import Optional

from agent_trust_lab.models.trap import EnhancedTrapDef
from agent_trust_lab.traps.generators import GENERATORS

_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")


class FieldMutator:
    """Applies field-level and template-interpolation mutations to trap definitions.

    Two modes:
    1. Template interpolation: if the field value contains ``{{generator_name}}``
       patterns, each placeholder is replaced with the corresponding generator output.
       The ``VariationRule.generator`` is ignored in this mode.
    2. Field-level replacement: if no templates are found, the entire field value
       is replaced with the output of ``VariationRule.generator`` (legacy mode).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def mutate(self, trap: EnhancedTrapDef, seed: Optional[int] = None) -> EnhancedTrapDef:
        """Create a mutated copy of the trap based on its variation_rules."""
        if seed is not None:
            self.rng = random.Random(seed)

        if not trap.variation_rules:
            return copy.deepcopy(trap)

        mutated = copy.deepcopy(trap)

        for rule in trap.variation_rules:
            if not hasattr(mutated, rule.field):
                continue
            current = getattr(mutated, rule.field)
            if isinstance(current, str) and _TEMPLATE_RE.search(current):
                new_value = self._interpolate(current)
                setattr(mutated, rule.field, new_value)
            else:
                new_value = self._generate(rule.generator)
                setattr(mutated, rule.field, new_value)

        return mutated

    def _interpolate(self, text: str) -> str:
        def _replacer(match: re.Match) -> str:
            return self._generate(match.group(1))

        return _TEMPLATE_RE.sub(_replacer, text)

    def _generate(self, generator: str) -> str:
        """Dispatch to the appropriate generator function."""
        handler = GENERATORS.get(generator)
        if handler is not None:
            return handler(self.rng)
        return self._default_generator()

    def _default_generator(self) -> str:
        return f"mutated_{uuid.uuid4().hex[:8]}"
