from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceReport:
    dimensions: Dict[str, str]  # dimension_name: pass/fail/warn
    critical_count: int = 0
    high_count: int = 0
    evidence: List[str] = field(default_factory=list)
    false_positive_events: List[str] = field(default_factory=list)
    benign_refusal_rate: Optional[float] = None

    def overall_status(self) -> str:
        if self.critical_count > 0:
            return "fail"
        if self.high_count >= 2:
            return "warn"
        return "pass"


@dataclass
class CodeHalluReport:
    step_index: int
    hallucination_type: str  # mapping/naming/parameter/logic_hallucination
    code_snippet: str
    error_message: Optional[str] = None
    expected_error_pattern: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class HalluStepReport:
    step_index: int
    gsar_label: str  # Grounded/Ungrounded/Contradicted/Complementary
    g_score: float = 0.0
    u_score: float = 0.0
    c_score: float = 0.0
    faithfulness_score: float = 1.0
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    anchor_type: str = "none"  # semantic/token_overlap/multi_hop/none


@dataclass
class CalibrationStatement:
    benchmark: str
    version: str
    kappa: float
    kappa_ci: tuple
    step_accuracy: float


@dataclass
class EvaluationReport:
    meta: Dict[str, Any] = field(default_factory=dict)
    compliance: Optional[ComplianceReport] = None
    hallucination_steps: List[HalluStepReport] = field(default_factory=list)
    code_agent_checks: List[CodeHalluReport] = field(default_factory=list)
    calibration_statement: Optional[CalibrationStatement] = None
    remediation: List[Dict[str, str]] = field(default_factory=list)
