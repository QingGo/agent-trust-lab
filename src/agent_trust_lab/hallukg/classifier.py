from typing import Any, Dict, List

from agent_trust_lab.models.report import HalluStepReport


class GSARClassifier:
    """Classify each step in a trajectory using GSAR labels (Grounded/Ungrounded/
    Contradicted/Complementary).

    Stub: returns mock GSAR labels — no real LLM classification.
    """

    def classify(
        self,
        steps: list,
        anchored_triples: List[Dict[str, Any]],
    ) -> List[HalluStepReport]:
        reports: List[HalluStepReport] = []
        gsar_labels = ["Grounded", "Grounded", "Complementary", "Ungrounded", "Contradicted"]

        for i, step in enumerate(steps):
            label = gsar_labels[i % len(gsar_labels)]
            g_score = 0.7 if label == "Grounded" else 0.3
            u_score = 0.1 if label != "Ungrounded" else 0.8
            c_score = 0.2 if label != "Contradicted" else 0.9

            reports.append(
                HalluStepReport(
                    step_index=i,
                    gsar_label=label,
                    g_score=g_score,
                    u_score=u_score,
                    c_score=c_score,
                    faithfulness_score=0.95,
                    evidence=["Mock triples anchoring evidence"],
                    explanation=f"Step {i}: Classified as {label} (stub)",
                )
            )

        return reports
