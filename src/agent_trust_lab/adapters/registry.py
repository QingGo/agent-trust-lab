"""Agent adapter registry with @register_adapter decorator.

Provides automatic discovery and retrieval of agent harness and sandbox
backend implementations. Any class decorated with @register_adapter
is added to the central registry.
"""

from typing import Dict, List, Optional, Type

from agent_trust_lab.log import get_logger
from agent_trust_lab.models.trajectory import AgentHarness

logger = get_logger("adapters.registry")

_registry: Dict[str, Type[AgentHarness]] = {}


def register_adapter(name: str):
    """Decorator to register an agent harness or sandbox backend class.

    Args:
        name: Unique adapter name (e.g. "langchain", "openai", "docker").

    The decorated class may optionally implement:

        @classmethod
        def from_config(cls, config: EvaluationConfig) -> AgentHarness: ...

    which allows the registry's resolve() to construct instances directly.
    """

    def decorator(cls: Type[AgentHarness]) -> Type[AgentHarness]:
        if name in _registry:
            logger.warning("Adapter '%s' is being re-registered (was %s)", name, _registry[name])
        _registry[name] = cls
        return cls

    return decorator


def get_adapter_class(name: str) -> Optional[Type[AgentHarness]]:
    """Retrieve a registered adapter class by name.

    Args:
        name: The adapter name used during registration.

    Returns:
        The registered class, or None if not found.
    """
    return _registry.get(name)


def list_adapters() -> List[str]:
    """Return sorted list of all registered adapter names."""
    return sorted(_registry.keys())


def resolve(name: str):
    """Resolve an adapter by name — returns the registered class.

    Args:
        name: Adapter name (e.g. "langchain").

    Returns:
        The registered AgentHarness subclass, or None if not found.
    """
    return _registry.get(name)
