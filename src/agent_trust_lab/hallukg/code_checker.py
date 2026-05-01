from typing import List, Optional

from agent_trust_lab.models.report import CodeHalluReport
from agent_trust_lab.models.trajectory import SecureTrajectory


class CodeHalluChecker:
    """Execute generated code in a sandbox to detect code hallucinations.

    Stub: returns mock CodeHalluReport — no real sandbox execution.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def check(
        self,
        code: str,
        test_command: str,
        expected_error: Optional[str] = None,
    ) -> CodeHalluReport:
        return CodeHalluReport(
            step_index=0,
            hallucination_type="naming",
            code_snippet=code[:100],
            error_message="Mock: ImportError for fake package (stub)",
            expected_error_pattern=expected_error,
            fix_suggestion="Use the correct package name from the verified list.",
        )

    def batch_check(self, trajectory: SecureTrajectory) -> List[CodeHalluReport]:
        code_steps = [(i, s) for i, s in enumerate(trajectory.steps) if s.type == "code_generation"]
        reports: List[CodeHalluReport] = []
        types = ["mapping", "naming", "parameter", "logic_hallucination"]
        for idx, (i, step) in enumerate(code_steps):
            reports.append(
                CodeHalluReport(
                    step_index=i,
                    hallucination_type=types[idx % len(types)],
                    code_snippet=step.content[:100],
                    error_message="Mock: code hallucination detected (stub)",
                    expected_error_pattern=None,
                    fix_suggestion=(
                        "Verify library name and parameter schema before code generation."
                    ),
                )
            )
        return reports
