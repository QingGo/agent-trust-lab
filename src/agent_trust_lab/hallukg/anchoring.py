import os
from typing import Any, Dict, List, Optional

import numpy as np

from agent_trust_lab.config import ONNX_CACHE_DIR
from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.anchoring")

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_CACHE = ONNX_CACHE_DIR

_GROUNDED_THRESHOLD = 0.3


class EmbeddingEngine:
    """Loads and runs an ONNX sentence embedding model.

    Lazy-loads on first use. Falls back to None if unavailable.
    """

    def __init__(self, model_name: str = _EMBED_MODEL_NAME, cache_dir: str = _DEFAULT_CACHE):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._session = None
        self._tokenizer = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        model_dir = os.path.join(self._cache_dir, self._model_name.replace("/", "_"))
        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            logger.debug("Embedding ONNX model not cached, using token overlap fallback")
            return

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(model_path)
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.debug("Embedding engine loaded: %s", model_dir)
        except Exception as e:
            logger.warning("Failed to load embedding ONNX model: %s", e)

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._session is not None and self._tokenizer is not None

    def encode(self, text: str) -> np.ndarray | None:
        if not self.is_available:
            return None
        tokenizer = self._tokenizer
        session = self._session
        if tokenizer is None or session is None:
            return None
        try:
            enc = tokenizer.encode(text)
            input_ids = np.array([enc.ids], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask], dtype=np.int64)
            feed_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = session.run(None, feed_dict)
            vec = np.asarray(outputs[0])[0].astype(np.float64)
            norm = np.linalg.norm(vec) + 1e-8
            return vec / norm
        except Exception as e:
            logger.warning("Embedding inference failed: %s", e)
            return None


import threading

_embedding_engine: Optional[EmbeddingEngine] = None
_engine_lock = threading.Lock()


def _get_embedding_engine() -> EmbeddingEngine:
    """Thread-safe lazy singleton — returns shared EmbeddingEngine instance."""
    global _embedding_engine
    if _embedding_engine is None:
        with _engine_lock:
            if _embedding_engine is None:
                _embedding_engine = EmbeddingEngine()
    return _embedding_engine


class AnchoringReasoner:
    """Anchor extracted triples against a knowledge base.

    Multi-tier grounding:
    1. ONNX semantic embedding → cosine similarity (most accurate)
    2. Token overlap matching (fallback, zero-dependency)
    3. Stub all-"Grounded" (when knowledge_text is empty)
    """

    def __init__(
        self,
        knowledge_base_path: str = "./kb/",
        code_index_path: Optional[str] = None,
        grounded_threshold: float = 0.3,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.code_index_path = code_index_path
        self.grounded_threshold = grounded_threshold
        self._embedder = _get_embedding_engine()

    def anchor(self, triple: Dict[str, Any], knowledge_text: str = "") -> Dict[str, Any]:
        subject = str(triple.get("subject", ""))
        predicate = str(triple.get("predicate", ""))
        obj = str(triple.get("object", ""))
        confidence = triple.get("confidence", 0.0)

        score, evidence, label = self._compute_anchor(subject, predicate, obj, knowledge_text)

        result: Dict[str, Any] = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": label,
            "evidence": evidence,
            "anchor_score": round(score, 4),
        }
        return result

    def batch_anchor(
        self, triples: List[Dict[str, Any]], knowledge_text: str = ""
    ) -> List[Dict[str, Any]]:
        if not triples:
            return []
        kb_sentences = self._split_knowledge(knowledge_text)
        return [self._anchor_single(t, kb_sentences, knowledge_text) for t in triples]

    def _anchor_single(
        self,
        triple: Dict[str, Any],
        kb_sentences: List[str],
        knowledge_text: str,
    ) -> Dict[str, Any]:
        subject = str(triple.get("subject", ""))
        predicate = str(triple.get("predicate", ""))
        obj = str(triple.get("object", ""))
        confidence = triple.get("confidence", 0.0)
        triple_text = " ".join(p for p in [subject, predicate, obj] if p)

        result = self._compute_anchor_semantic(triple_text, kb_sentences, knowledge_text)
        if result is not None:
            score, evidence, label = result
        else:
            score, evidence, label = self._compute_anchor_token_overlap(
                subject, obj, knowledge_text
            )

        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": label,
            "evidence": evidence,
            "anchor_score": round(score, 4),
        }

    def _compute_anchor(
        self,
        subject: str,
        predicate: str,
        obj: str,
        knowledge_text: str,
    ) -> tuple[float, List[str], str]:
        kb_sentences = self._split_knowledge(knowledge_text)
        triple_text = " ".join(p for p in [subject, predicate, obj] if p)

        result = self._compute_anchor_semantic(triple_text, kb_sentences, knowledge_text)
        if result is not None:
            return result

        return self._compute_anchor_token_overlap(subject, obj, knowledge_text)

    def _compute_anchor_semantic(
        self,
        triple_text: str,
        kb_sentences: list[str],
        knowledge_text: str,
    ) -> Optional[tuple[float, list[str], str]]:
        if not knowledge_text:
            return None
        if not kb_sentences:
            return None

        triple_emb = self._embedder.encode(triple_text)
        if triple_emb is None:
            return None

        best_score = -1.0
        best_sentence = ""
        for sentence in kb_sentences:
            kb_emb = self._embedder.encode(sentence)
            if kb_emb is None:
                continue
            sim = float(np.dot(triple_emb, kb_emb))
            if sim > best_score:
                best_score = sim
                best_sentence = sentence

        if best_score < 0:
            return None

        grounded = best_score >= self.grounded_threshold
        label = "Grounded" if grounded else "Ungrounded"
        evidence = (
            [f"Semantic match ({best_score:.3f}) with: '{best_sentence}'"]
            if grounded
            else [f"No semantic match (max {best_score:.3f}) against knowledge source"]
        )
        return best_score, evidence, label

    def _compute_anchor_token_overlap(
        self, subject: str, obj: str, knowledge_text: str
    ) -> tuple[float, List[str], str]:
        if not knowledge_text:
            return 0.92, ["Stub: no knowledge source provided (all grounded)"], "Grounded"

        lower_kb = knowledge_text.lower()
        matches = []
        if subject and subject.lower() in lower_kb:
            matches.append(subject)
        if obj and obj.lower() in lower_kb:
            matches.append(obj)

        if matches:
            score = 0.92
            label = "Grounded" if score >= self.grounded_threshold else "Ungrounded"
            return (
                score,
                [f"Token match for '{', '.join(matches)}' in knowledge source"],
                label,
            )
        score = 0.15
        label = "Grounded" if score >= self.grounded_threshold else "Ungrounded"
        return (
            score,
            [f"No token match for '{subject}' or '{obj}' in knowledge source"],
            label,
        )

    @staticmethod
    def _split_knowledge(knowledge_text: str) -> List[str]:
        if not knowledge_text:
            return []
        sentences = []
        for part in knowledge_text.replace("\n", ". ").split(". "):
            stripped = part.strip()
            if stripped:
                sentences.append(stripped)
        return sentences
