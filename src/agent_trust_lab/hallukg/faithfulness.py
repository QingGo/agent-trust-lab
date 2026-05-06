from typing import Any, Optional

from agent_trust_lab.config import ONNX_CACHE_DIR
from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.faithfulness")

_nli_session: Optional[Any] = None
_nli_tokenizer: Optional[Any] = None
_nli_loaded: bool = False


def _ensure_nli_loaded(nli_neutral_weight: float = 0.5) -> bool:
    """Lazy-load ONNX NLI session and tokenizer once at module level.

    Returns True if ONNX NLI is available.
    """
    global _nli_session, _nli_tokenizer, _nli_loaded
    if _nli_loaded:
        return _nli_session is not None and _nli_tokenizer is not None
    _nli_loaded = True

    try:
        import onnxruntime as _ort  # noqa: F401
    except ImportError:
        logger.debug("onnxruntime not available, using TF-IDF fallback")
        return False

    import os

    model_path = os.path.join(ONNX_CACHE_DIR, "roberta-base-mnli", "model.onnx")
    tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        logger.debug("ONNX NLI model not cached, using TF-IDF fallback")
        return False

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        _nli_session = ort.InferenceSession(model_path)
        _nli_tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.debug("ONNX NLI engine loaded")
        return True
    except Exception as e:
        logger.warning("Failed to load ONNX NLI model: %s", e)
        return False


class FaithfulnessChecker:
    """Check agent output statements against evidence for faithfulness.

    Primary: TF-IDF cosine similarity (zero-dependency, deterministic).
    Optional: ONNX NLI via roberta-base-mnli when onnxruntime is available
    and the ONNX model has been exported to the local cache.

    ONNX model is loaded once at module level and reused across all instances.
    """

    def __init__(self, nli_neutral_weight: float = 0.5) -> None:
        self.nli_neutral_weight = nli_neutral_weight
        self._onnx_available = _ensure_nli_loaded(nli_neutral_weight)

    @staticmethod
    def _run_nli_inference(
        session: Any,
        tokenizer: Any,
        premise: str,
        hypothesis: str,
    ) -> Any:
        import numpy as np

        encoded = tokenizer.encode(premise, hypothesis)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        input_feed = {}
        input_names = [i.name for i in session.get_inputs()]
        if "input_ids" in input_names:
            input_feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            input_feed["attention_mask"] = attention_mask

        outputs = session.run(None, input_feed)
        return np.asarray(outputs[0])[0]

    def check(self, statements: list[str], evidence: list[str]) -> float:
        statement_text = " ".join(statements).strip()
        evidence_text = " ".join(evidence).strip()

        if not statement_text or not evidence_text:
            return 0.5

        if self._onnx_available:
            score = self._onnx_nli(evidence_text, statement_text)
            if score is not None:
                return score

        return self._tfidf_similarity(evidence_text, statement_text)

    def batch_check(
        self,
        statement_batches: list[list[str]],
        evidence_batches: list[list[str]],
    ) -> list[float]:
        n = len(statement_batches)
        if n == 0:
            return []

        statement_texts = [" ".join(stmts).strip() for stmts in statement_batches]
        evidence_texts = [" ".join(evids).strip() for evids in evidence_batches]

        results: list[float] = []
        for stmt, evid in zip(statement_texts, evidence_texts):
            if not stmt or not evid:
                results.append(0.5)
                continue
            if self._onnx_available:
                score = self._onnx_nli(evid, stmt)
                if score is not None:
                    results.append(score)
                    continue
            results.append(self._tfidf_similarity(evid, stmt))
        return results

    def _tfidf_similarity(self, text_a: str, text_b: str) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("scikit-learn not available, returning neutral score")
            return 0.5

        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return round(float(sim), 4)
        except Exception as e:
            logger.warning("TF-IDF similarity failed: %s", e)
            return 0.5

    def _onnx_nli(self, premise: str, hypothesis: str) -> float | None:
        if _nli_session is None or _nli_tokenizer is None:
            return None

        try:
            import numpy as np

            logits = self._run_nli_inference(
                _nli_session, _nli_tokenizer, premise, hypothesis
            )

            exp_x = np.exp(logits - np.max(logits))
            probs = exp_x / exp_x.sum()

            score = float(
                probs[2] * 1.0 + probs[1] * self.nli_neutral_weight + probs[0] * 0.0
            )
            return round(score, 4)
        except Exception as e:
            logger.warning("ONNX NLI inference failed: %s", e)
            return None
