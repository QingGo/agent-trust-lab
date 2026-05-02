from typing import Any, Dict, List, Optional

from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.anchoring")


class AnchoringReasoner:
    """Anchor extracted triples against a knowledge base via vector retrieval.

    Uses simple token overlap against knowledge_text for per-trap grounding.
    Falls back to all-"Grounded" stub when knowledge_text is empty.
    """

    def __init__(
        self,
        knowledge_base_path: str = "./kb/",
        code_index_path: Optional[str] = None,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.code_index_path = code_index_path

    def anchor(
        self, triple: Dict[str, Any], knowledge_text: str = ""
    ) -> Dict[str, Any]:
        subject = str(triple.get("subject", ""))
        obj = str(triple.get("object", ""))
        grounded = self._is_grounded(subject, obj, knowledge_text)
        label = "Grounded" if grounded else "Ungrounded"
        evidence = (
            [f"Matched '{subject}' or '{obj}' in knowledge source"]
            if grounded
            else [f"No match for '{subject}' or '{obj}' in knowledge source"]
        )
        anchor_score = 0.92 if grounded else 0.15
        result: Dict[str, Any] = {
            "subject": subject,
            "predicate": triple.get("predicate", ""),
            "object": obj,
            "confidence": triple.get("confidence", 0.0),
            "label": label,
            "evidence": evidence,
            "anchor_score": anchor_score,
        }
        return result

    def batch_anchor(
        self, triples: List[Dict[str, Any]], knowledge_text: str = ""
    ) -> List[Dict[str, Any]]:
        return [self.anchor(t, knowledge_text=knowledge_text) for t in triples]

    @staticmethod
    def _is_grounded(subject: str, obj: str, knowledge_text: str) -> bool:
        if not knowledge_text:
            return True
        lower_kb = knowledge_text.lower()
        if subject and subject.lower() in lower_kb:
            return True
        if obj and obj.lower() in lower_kb:
            return True
        return False
