"""Backward-compat shim — delegates to agent_trust_lab.pipeline.

All existing consumers using:

    from agent_trust_lab.orchestrator import Orchestrator, EvaluationResult

continue to work unchanged.
"""

from agent_trust_lab.pipeline import EvaluationResult, Orchestrator  # noqa: F401

__all__ = ["EvaluationResult", "Orchestrator"]
