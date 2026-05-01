from typing import List


class FaithfulnessChecker:
    """Compare agent output text against observations for faithfulness.

    Stub: returns mock faithfulness scores — no real NLI model.
    """

    def check(self, statements: List[str], evidence: List[str]) -> float:
        return 0.95

    def batch_check(
        self,
        statement_batches: List[List[str]],
        evidence_batches: List[List[str]],
    ) -> List[float]:
        return [0.95] * len(statement_batches)
