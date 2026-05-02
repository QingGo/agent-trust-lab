from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    type: str
    content: str
    tools_called: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    event_type: str
    description: str
    step_index: int


@dataclass
class SecureTrajectory:
    steps: List[TrajectoryStep]
    security_events: List[SecurityEvent]
    dry_run_log: str = ""
    policy_rules_applied: List[str] = field(default_factory=list)
    actual_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "type": s.type,
                    "content": s.content,
                    "tools_called": s.tools_called,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "security_events": [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "step_index": e.step_index,
                }
                for e in self.security_events
            ],
            "dry_run_log": self.dry_run_log,
            "policy_rules_applied": self.policy_rules_applied,
            "actual_violations": self.actual_violations,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecureTrajectory":
        return cls(
            steps=[TrajectoryStep(**s) for s in data.get("steps", [])],
            security_events=[
                SecurityEvent(**e) for e in data.get("security_events", [])
            ],
            dry_run_log=data.get("dry_run_log", ""),
            policy_rules_applied=data.get("policy_rules_applied", []),
            actual_violations=data.get("actual_violations", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentHarness(ABC):
    @abstractmethod
    def run(
        self,
        task: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 10,
        policy_rules: Optional[List[str]] = None,
    ) -> SecureTrajectory: ...
