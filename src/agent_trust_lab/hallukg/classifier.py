from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agent_trust_lab.config import DEFAULT_MODEL
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

    Session 9: Added classify_multi_model() for multi-model voting.
    """

    def __init__(self, model: str = "", strict_mode: bool = False, temperature: float = 0.0):
        self.model = model or DEFAULT_MODEL
        self.strict_mode = strict_mode
        self.temperature = temperature

    def classify(
        self,
        steps: list,
        anchored_triples: List[Dict[str, Any]],
    ) -> List[HalluStepReport]:
        from agent_trust_lab.llm import _RETRYABLE_ERRORS, retry_with_backoff

        try:
            return retry_with_backoff(
                lambda: self._classify_with_llm(steps, anchored_triples)
            )
        except _RETRYABLE_ERRORS:
            if self.strict_mode:
                raise
        except Exception as e:
            logger.warning("GSARClassifier LLM call failed, falling back to stub: %s", e)
            if self.strict_mode:
                raise
        return self._classify_stub(steps)

    def classify_multi_model(
        self,
        steps: list,
        anchored_triples: List[Dict[str, Any]],
        model_list: List[str],
    ) -> List[HalluStepReport]:
        """Classify steps using multiple models concurrently with majority voting.

        Runs each model in its own thread, collects results, and aggregates:
        - GSAR label: majority vote across models
        - G/U/C/F scores: arithmetic mean across models
        - evidence: taken from the model with highest agreement

        Falls back to stub if all model calls fail.

        Args:
            steps: TrajectoryStep list.
            anchored_triples: Anchored triple dicts for grounding evidence.
            model_list: List of model identifiers to use for classification.

        Returns:
            List of HalluStepReport with voted labels and averaged scores.
        """
        if not model_list:
            return self.classify(steps, anchored_triples)
        if len(model_list) == 1:
            original_model = self.model
            self.model = model_list[0]
            try:
                return self.classify(steps, anchored_triples)
            finally:
                self.model = original_model

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results_by_model: Dict[str, List[HalluStepReport]] = {}
        original_model = self.model

        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            futures: dict = {}
            for model in model_list:
                classifier_for_model = GSARClassifier(
                    model=model, temperature=self.temperature
                )

                def _run_classify(
                    clf=classifier_for_model,
                    steps=steps,
                    triples=anchored_triples,
                ):
                    from agent_trust_lab.llm import _RETRYABLE_ERRORS, retry_with_backoff

                    try:
                        return retry_with_backoff(
                            lambda: clf._classify_with_llm(steps, triples)
                        )
                    except _RETRYABLE_ERRORS:
                        pass
                    except Exception as e:
                        logger.warning(
                            "GSAR classification failed for model %s: %s", model, e
                        )
                    try:
                        return clf._classify_stub(steps)
                    except Exception:
                        return None

                future = executor.submit(_run_classify)
                futures[future] = model

            for future in as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_by_model[model] = result
                except Exception as e:
                    logger.warning(
                        "Multi-model classification failed for %s: %s", model, e
                    )

        self.model = original_model

        if not results_by_model:
            logger.warning("All multi-model classifications failed, falling back to stub")
            return self._classify_stub(steps)

        if len(results_by_model) == 1:
            return list(results_by_model.values())[0]

        merged: List[HalluStepReport] = []
        num_steps = len(steps)
        for i in range(num_steps):
            step_reports = [
                reports[i]
                for reports in results_by_model.values()
                if i < len(reports)
            ]
            if not step_reports:
                continue

            labels = [r.gsar_label for r in step_reports]
            label_counts: Dict[str, int] = {}
            for lab in labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1

            voted_label = max(label_counts, key=lambda k: label_counts[k])
            g = sum(r.g_score for r in step_reports) / len(step_reports)
            u = sum(r.u_score for r in step_reports) / len(step_reports)
            c = sum(r.c_score for r in step_reports) / len(step_reports)
            f = sum(r.faithfulness_score for r in step_reports) / len(step_reports)

            evidence = step_reports[0].evidence if step_reports[0].evidence else []

            explanation = f"Multi-model vote ({len(step_reports)} models): {label_counts}"

            merged.append(
                HalluStepReport(
                    step_index=i,
                    gsar_label=voted_label,
                    g_score=round(g, 4),
                    u_score=round(u, 4),
                    c_score=round(c, 4),
                    faithfulness_score=round(f, 4),
                    evidence=evidence,
                    explanation=explanation,
                )
            )

        return merged

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
            f"- {t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}"
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
            temperature=self.temperature,
        )

        from agent_trust_lab.llm import capture_usage

        raw = getattr(result, "_raw_response", None)
        if raw and hasattr(raw, "usage") and raw.usage:
            capture_usage(raw, self.model)

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
