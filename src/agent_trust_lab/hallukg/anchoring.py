from typing import Any, Dict, List, Optional


class AnchoringReasoner:
    """Anchor extracted triples against a knowledge base via vector retrieval.

    Stub: returns mock grounding labels — no real KB or vector search.
    """

    def __init__(
        self,
        knowledge_base_path: str = "./kb/",
        code_index_path: Optional[str] = None,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.code_index_path = code_index_path

    def anchor(self, triple: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "label": "Grounded",
            "evidence": ["Mock evidence from knowledge base"],
            "score": 0.92,
        }

    def batch_anchor(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.anchor(t) for t in triples]
