from typing import List

from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.faithfulness")


class FaithfulnessChecker:
    """Check agent output statements against evidence for faithfulness.

    Primary: TF-IDF cosine similarity (zero-dependency, deterministic).
    Optional: ONNX NLI via roberta-base-mnli when onnxruntime is available
    and the ONNX model has been exported to the local cache.
    """

    def __init__(self, nli_neutral_weight: float = 0.5) -> None:
        self.nli_neutral_weight = nli_neutral_weight
        self._onnx_available = self._check_onnx()

    @staticmethod
    def _check_onnx() -> bool:
        try:
            import onnxruntime as _ort  # noqa: F401
        except ImportError:
            logger.debug("onnxruntime not available, using TF-IDF fallback")
            return False
        logger.debug("onnxruntime available, ONNX NLI ready")
        return True

    def check(self, statements: List[str], evidence: List[str]) -> float:
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
        statement_batches: List[List[str]],
        evidence_batches: List[List[str]],
    ) -> List[float]:
        n = len(statement_batches)
        if n == 0:
            return []

        statement_texts = [" ".join(stmts).strip() for stmts in statement_batches]
        evidence_texts = [" ".join(evids).strip() for evids in evidence_batches]

        results: List[float] = []
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
        try:
            import os

            import numpy as np

            model_path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "agent-trust-lab",
                "onnx",
                "roberta-base-mnli",
                "model.onnx",
            )
            tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer.json")

            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                logger.debug("ONNX model not cached, skipping ONNX NLI")
                return None

            import onnxruntime as ort
            from tokenizers import Tokenizer

            session = ort.InferenceSession(model_path)
            tokenizer = Tokenizer.from_file(tokenizer_path)

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
            logits = np.asarray(outputs[0])[0]

            exp_x = np.exp(logits - np.max(logits))
            probs = exp_x / exp_x.sum()

            score = float(probs[2] * 1.0 + probs[1] * self.nli_neutral_weight + probs[0] * 0.0)
            return round(score, 4)
        except Exception as e:
            logger.warning("ONNX NLI inference failed: %s", e)
            return None
