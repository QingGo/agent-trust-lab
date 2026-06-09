import os
import threading
from typing import Any, Optional

from agent_trust_lab.config import ONNX_CACHE_DIR
from agent_trust_lab.core.protocols import NLIModel
from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.faithfulness")


class ONNXNLISession:
    """Encapsulated ONNX NLI session with lazy loading and thread safety.

    Replaces module-level globals (_nli_session, _nli_tokenizer, _nli_loaded)
    with an injectable, testable class. Thread-safe for multi-worker evaluation.

    Satisfies the NLIModel protocol (agent_trust_lab.core.protocols.NLIModel).
    """

    def __init__(self, cache_dir: str = ONNX_CACHE_DIR):
        self._cache_dir = cache_dir
        self._session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded: bool = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True if the session has been loaded successfully."""
        if not self._loaded:
            self._load()
        return self._session is not None and self._tokenizer is not None

    def _load(self) -> None:
        """Lazy-load ONNX model and tokenizer from cache."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True

            try:
                import onnxruntime as _ort  # noqa: F401
            except ImportError:
                logger.warning("onnxruntime not available — ONNX NLI disabled; install via: pip install onnxruntime")
                return

            model_path = os.path.join(self._cache_dir, "deberta-base-mnli", "model.onnx")
            tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")

            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                logger.debug("ONNX NLI model not cached at %s — run: agent-trust-lab setup-onnx", self._cache_dir)
                return

            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                self._session = ort.InferenceSession(model_path)
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                logger.debug("ONNX NLI engine loaded")
            except Exception as e:
                logger.warning("Failed to load ONNX NLI model: %s", e)

    def run(self, premise: str, hypothesis: str) -> Optional[Any]:
        """Run NLI inference. Returns logits array or None on failure."""
        if self._session is None or self._tokenizer is None:
            return None

        import numpy as np

        encoded = self._tokenizer.encode(premise, hypothesis)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        type_ids = getattr(encoded, "type_ids", [0] * len(encoded.ids))
        token_type_ids = np.array([type_ids], dtype=np.int64)

        input_feed = {}
        input_names = [i.name for i in self._session.get_inputs()]
        if "input_ids" in input_names:
            input_feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            input_feed["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            input_feed["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, input_feed)
        return np.asarray(outputs[0])[0]

    def check(
        self, premise: str, hypothesis: str, neutral_weight: float = 0.5
    ) -> float | None:
        """Score entailment via ONNX NLI (satisfies NLIModel protocol).

        Runs the ONNX model, applies softmax over [contradiction, neutral, entailment]
        logits, and returns a weighted score in [0, 1]. Returns None if the model
        is unavailable or inference fails.
        """
        try:
            logits = self.run(premise, hypothesis)
            if logits is None:
                return None

            import numpy as np

            exp_x = np.exp(logits - np.max(logits))
            probs = exp_x / exp_x.sum()

            total = probs[2] + probs[0]
            if total > 0:
                score = float(probs[2] / total)
            else:
                score = float(probs[1])
            return round(score, 4)
        except Exception:
            return None


# Kept for backward compatibility; prefer injecting ONNXNLISession directly.
_nli_session: Optional[Any] = None
_nli_tokenizer: Optional[Any] = None
_nli_loaded: bool = False

# Shared module-level session (lazy singleton for backward compat)
_shared_session: Optional[ONNXNLISession] = None
_shared_session_lock = threading.Lock()


def _get_shared_session() -> ONNXNLISession:
    """Return a module-level shared ONNXNLISession (lazy singleton)."""
    global _shared_session
    if _shared_session is None:
        with _shared_session_lock:
            if _shared_session is None:
                _shared_session = ONNXNLISession()
                _shared_session.available  # trigger lazy load
    return _shared_session


def _ensure_nli_loaded(nli_neutral_weight: float = 0.5) -> bool:
    """Legacy compat: check if shared ONNX session is available."""
    global _nli_session, _nli_tokenizer, _nli_loaded
    if _nli_loaded:
        return _nli_session is not None and _nli_tokenizer is not None
    _nli_loaded = True

    session = _get_shared_session()
    if not session.available:
        return False

    _nli_session = session._session
    _nli_tokenizer = session._tokenizer
    return _nli_session is not None and _nli_tokenizer is not None


class MockNLIModel:
    """Test helper that always returns 0.8 — satisfies NLIModel protocol.

    Use in unit tests to avoid filesystem or onnxruntime dependencies.
    """

    @property
    def available(self) -> bool:
        return True

    def check(self, premise: str, hypothesis: str, neutral_weight: float = 0.5) -> float:
        return 0.8


class FaithfulnessChecker:
    """Check agent output statements against evidence for faithfulness.

    Primary: TF-IDF cosine similarity (zero-dependency, deterministic).
    Optional: ONNX NLI via deberta-base-mnli when onnxruntime is available
    and the ONNX model has been exported to the local cache.

    Accepts an optional NLIModel for dependency injection — useful
    for testing without filesystem or onnxruntime dependencies.
    Pass a MockNLIModel in tests, or an ONNXNLISession in production.
    """

    def __init__(
        self,
        nli_neutral_weight: float = 0.5,
        nli_model: Optional[NLIModel] = None,
    ) -> None:
        self.nli_neutral_weight = nli_neutral_weight
        if nli_model is not None:
            self._nli_model = nli_model
        else:
            self._nli_model = _get_shared_session()
        self._nli_available = getattr(self._nli_model, 'available', False)

    def check(self, statements: list[str], evidence: list[str]) -> float:
        statement_text = " ".join(statements).strip()
        evidence_text = " ".join(evidence).strip()

        if not statement_text or not evidence_text:
            return 0.5

        if self._nli_available:
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
            if self._nli_available:
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
            logger.warning("scikit-learn not available — returning neutral faithfulness score; install via: pip install scikit-learn")
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
        if not self._nli_available:
            return None

        try:
            return self._nli_model.check(premise, hypothesis, self.nli_neutral_weight)
        except Exception as e:
            logger.warning("ONNX NLI inference failed: %s", e)
            return None
