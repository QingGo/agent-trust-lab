"""Orchestrator pipeline package.

Provides the Orchestrator class and EvaluationResult dataclass.
"""

from agent_trust_lab.pipeline.models import EvaluationResult
from agent_trust_lab.pipeline.orchestrator import Orchestrator

__all__ = ["EvaluationResult", "Orchestrator"]
