from dataclasses import dataclass
from typing import Dict, List, Optional, Type


@dataclass
class InjectionTemplate:
    name: str = ""
    description: str = ""

    def render(self, base_task: str, injection: str, tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError


_registry: Dict[str, Type[InjectionTemplate]] = {}


def register_template(name: str):
    def decorator(cls: Type[InjectionTemplate]) -> Type[InjectionTemplate]:
        _registry[name] = cls
        return cls
    return decorator


def resolve_template(name: str) -> Type[InjectionTemplate]:
    if name not in _registry:
        from agent_trust_lab.traps.templates.system_note import SystemNoteTemplate
        return SystemNoteTemplate
    return _registry[name]


from agent_trust_lab.traps.templates import (  # noqa: E402,F401
    code_comment,
    colleague,
    documentation,
    hidden_in_context,
    system_note,
)
