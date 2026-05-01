from typing import Any, Dict, List


class TripleExtractor:
    """Extract {subject, predicate, object, confidence} triples from text.

    Stub: returns mock triples — no real LLM calls.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def extract(self, text: str) -> List[Dict[str, Any]]:
        return [
            {
                "subject": "agent",
                "predicate": "generated_output",
                "object": text[:80].strip().replace("\n", " "),
                "confidence": 0.85,
            }
        ]
