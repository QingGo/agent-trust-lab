from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import HalluStepReport

logger = get_logger("hallukg.classifier")


class GSARStepResult(BaseModel):
    step_index: int = Field(ge=0)
    gsar_label: str = Field(description="Grounded/Ungrounded/Contradicted/Complementary")
    g_score: float = Field(default=0.0, ge=0.0, le=1.0)
    u_score: float = Field(default=0.0, ge=0.0, le=1.0)
    c_score: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")


class GSAROutput(BaseModel):
    steps: List[GSARStepResult]


class GSARClassifier:
    """Classify each step in a trajectory using GSAR labels (Grounded/Ungrounded/
    Contradicted/Complementary).

    Uses real DeepSeek LLM calls via instructor for structured classification.
    Falls back to stub on any API error.
    """

    def __init__(self, model: str = ""):
        self.model = model or "deepseek-v4-flash"

    def classify(
        self,
        steps: list,
        anchored_triples: List[Dict[str, Any]],
    ) -> List[HalluStepReport]:
        try:
            return self._classify_with_llm(steps, anchored_triples)
        except Exception as e:
            logger.warning("GSARClassifier LLM call failed, falling back to stub: %s", e)
            return self._classify_stub(steps)

    def _classify_with_llm(
        self,
        steps: list,
        anchored_triples: List[Dict[str, Any]],
    ) -> List[HalluStepReport]:
        import instructor

        from agent_trust_lab.llm import create_openai_client, get_api_key, get_base_url

        api_key = get_api_key()
        if not api_key:
            raise ValueError("No API key available for GSARClassifier LLM call")

        step_entries = []
        for i, step in enumerate(steps):
            content = getattr(step, "content", str(step))
            step_type = getattr(step, "type", "unknown")
            step_entries.append(f"Step {i} (type={step_type}): {content}")

        triples_text = "\n".join(
            f"- {t.get('subject','')} {t.get('predicate','')} {t.get('object','')}"
            for t in anchored_triples
        )

        client = create_openai_client(api_key=api_key, base_url=get_base_url())
        instructor_client = instructor.from_openai(client)

        result = instructor_client.chat.completions.create(
            model=self.model,
            response_model=GSAROutput,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hallucination classifier for AI agent outputs. "
                        "For each step below, classify it with ONE of these GSAR labels:\n"
                        "- Grounded: the step's claims are all supported by the anchored triples\n"
                        "- Ungrounded: the step makes claims NOT supported by any anchored triple\n"
                        "- Contradicted: the step directly contradicts anchored triples\n"
                        "- Complementary: adds extra useful context beyond the triples\n\n"
                        "Provide scores (0-1) for g_score, u_score, c_score. "
                        "Also provide faithfulness_score, evidence list, "
                        "and a brief explanation for each step."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Classify each of these agent trajectory steps:\n\n"
                        + "\n\n".join(step_entries)
                        + "\n\nAnchored knowledge triples:\n"
                        + (triples_text or "(no anchored triples available)")
                    ),
                },
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )

        reports: List[HalluStepReport] = []
        for s in result.steps:
            reports.append(
                HalluStepReport(
                    step_index=s.step_index,
                    gsar_label=s.gsar_label,
                    g_score=s.g_score,
                    u_score=s.u_score,
                    c_score=s.c_score,
                    faithfulness_score=s.faithfulness_score,
                    evidence=s.evidence,
                    explanation=s.explanation,
                )
            )
        return reports

    def _classify_stub(self, steps: list) -> List[HalluStepReport]:
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
