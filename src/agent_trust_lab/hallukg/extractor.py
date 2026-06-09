from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agent_trust_lab.config import DEFAULT_MODEL
from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.extractor")


class TripleEntry(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class TripleList(BaseModel):
    triples: List[TripleEntry]


class TripleExtractor:
    """Extract {subject, predicate, object, confidence} triples from text.

    Uses real DeepSeek LLM calls via instructor for structured output.
    Falls back to stub on any API error.
    """

    def __init__(
        self,
        model: str = "",
        strict_mode: bool = False,
        instructor_client: Optional[Any] = None,
    ):
        self.model = model or DEFAULT_MODEL
        self.strict_mode = strict_mode
        self._instructor_client = instructor_client

    def extract(self, text: str) -> List[Dict[str, Any]]:
        from agent_trust_lab.llm import _RETRYABLE_ERRORS, retry_with_backoff

        try:
            return retry_with_backoff(
                lambda: self._extract_with_llm(text)
            )
        except _RETRYABLE_ERRORS:
            if self.strict_mode:
                raise
        except Exception as e:
            logger.warning(
                "TripleExtractor LLM call failed after retries, "
                "falling back to stub: %s", e
            )
            if self.strict_mode:
                raise
        return self._extract_stub(text)

    def _extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        from agent_trust_lab.llm import (
            create_instructor_client,
            get_api_key,
            get_base_url,
        )

        api_key = get_api_key()
        if not api_key:
            raise ValueError("No API key available for TripleExtractor LLM call")

        instructor_client = self._instructor_client or create_instructor_client(
            api_key=api_key, base_url=get_base_url()
        )

        result = instructor_client.chat.completions.create(
            model=self.model,
            response_model=TripleList,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge graph triple extractor. Extract (subject, predicate, "
                        "object) information triples from the given text. Each triple must have a "
                        "subject, predicate, object, and confidence score between 0 and 1. "
                        "Output only factually grounded triples — do not hallucinate. "
                        "If the text contains no extractable facts, return an empty triples list."
                    ),
                },
                {"role": "user", "content": text},
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )

        from agent_trust_lab.llm import capture_usage

        raw = getattr(result, "_raw_response", None)
        if raw and hasattr(raw, "usage") and raw.usage:
            capture_usage(raw, self.model)

        return [t.model_dump() for t in result.triples]

    def _extract_stub(self, text: str) -> List[Dict[str, Any]]:
        """Return empty list on LLM failure — no fake triples."""
        logger.warning("TripleExtractor using stub: returning empty triples list for text[%d]", len(text))
        return []
