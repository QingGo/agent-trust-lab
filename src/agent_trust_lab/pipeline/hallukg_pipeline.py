"""Hallucination KG pipeline for orchestrator.

Handles triple extraction, evidence anchoring, multi-hop reasoning,
GSAR classification, faithfulness checking, and calibration.
"""

from typing import Any, Dict, List

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.report import CodeHalluReport, HalluStepReport
from agent_trust_lab.models.trajectory import SecureTrajectory
from agent_trust_lab.pipeline.sampling import run_adaptive_sampling, run_self_consistency

logger = get_logger("orchestrator")


class HalluKGPipeline:
    """Runs the full hallucination KG detection flow on a trajectory."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def run(
        self,
        trajectory: SecureTrajectory,
        is_code_agent: bool = False,
        knowledge_source: str = "",
    ) -> tuple[List[HalluStepReport], List[CodeHalluReport], Dict[str, Any]]:
        from agent_trust_lab.hallukg.anchoring import AnchoringReasoner
        from agent_trust_lab.hallukg.classifier import GSARClassifier
        from agent_trust_lab.hallukg.extractor import TripleExtractor
        from agent_trust_lab.hallukg.multi_hop import MultiHopReasoner

        cfg = self._config
        judge_model = cfg.judge_model or cfg.model
        extractor = TripleExtractor(model=judge_model, strict_mode=cfg.strict_mode)
        reasoner = AnchoringReasoner(
            knowledge_base_path=cfg.anchor_kb,
            grounded_threshold=cfg.grounded_threshold,
        )
        classifier = GSARClassifier(
            model=judge_model,
            strict_mode=cfg.strict_mode,
            temperature=cfg.temperature,
        )

        skip_types = set(cfg.skip_extract_types)
        all_triples = []
        step_anchor_types: dict[int, str] = {}
        raw_triples_by_step: dict[int, list] = {}
        for i, step in enumerate(trajectory.steps):
            if step.type == "trap_injection" or step.type in skip_types:
                continue
            triples = extractor.extract(step.content)
            raw_triples_by_step[i] = [
                dict(t) if isinstance(t, dict) else {"data": str(t)}
                for t in triples
            ]
            anchored = reasoner.batch_anchor(triples, knowledge_text=knowledge_source)
            all_triples.extend(anchored)
            step_anchor_types[i] = self._step_anchor_type(anchored)

        if knowledge_source:
            try:
                multi_hopper = MultiHopReasoner(
                    grounded_threshold=cfg.grounded_threshold,
                )
                multi_anchored = multi_hopper.batch_anchor(
                    all_triples, knowledge_text=knowledge_source
                )
                all_triples = self._merge_anchor_results(all_triples, multi_anchored)
                merged_anchor_type = self._step_anchor_type(all_triples)
                for idx in step_anchor_types:
                    step_anchor_types[idx] = merged_anchor_type
            except Exception as e:
                logger.warning(
                    "Multi-hop reasoning skipped (graph construction failed): %s", e
                )

        hallucination_steps = classifier.classify(trajectory.steps, all_triples)

        if cfg.gsar_vote_enabled and cfg.gsar_vote_models:
            try:
                multi_model_steps = classifier.classify_multi_model(
                    trajectory.steps, all_triples, cfg.gsar_vote_models
                )
                if multi_model_steps:
                    hallucination_steps = multi_model_steps
            except Exception as e:
                logger.warning(
                    "Multi-model GSAR voting failed, using single-model result: %s", e
                )

        for step in hallucination_steps:
            step.anchor_type = step_anchor_types.get(step.step_index, "none")

        self.apply_faithfulness_check(hallucination_steps, trajectory)
        code_hallus: List[CodeHalluReport] = []

        if cfg.adaptive_sampling and hallucination_steps:
            high_d_indices = run_adaptive_sampling(
                hallucination_steps, trajectory, classifier, all_triples, cfg
            )
            if high_d_indices:
                self.apply_faithfulness_check(
                    [hallucination_steps[i] for i in high_d_indices],
                    trajectory,
                )

        if cfg.self_consistency_enabled and hallucination_steps:
            run_self_consistency(
                hallucination_steps, trajectory, classifier, all_triples, cfg
            )
            self.apply_faithfulness_check(hallucination_steps, trajectory)

        if is_code_agent:
            from agent_trust_lab.hallukg.code_checker import CodeHalluChecker

            code_checker = CodeHalluChecker(
                timeout=cfg.timeout,
                docker_host=cfg.docker_host,
                strict_mode=cfg.strict_mode,
            )
            code_hallus = code_checker.batch_check(trajectory)

        checkpoint: Dict[str, Any] = {
            "anchored_triples": all_triples,
            "raw_triples_by_step": {str(k): v for k, v in raw_triples_by_step.items()},
            "step_anchor_types": step_anchor_types,
            "knowledge_source": knowledge_source,
        }
        return hallucination_steps, code_hallus, checkpoint

    def apply_calibration(self, hallucination_steps: List[HalluStepReport]) -> None:
        from agent_trust_lab.calibration.profile import load_profile

        profile_id = self._config.calibration_profile or ""
        profile = load_profile(profile_id)
        if profile is None:
            logger.warning(
                "Calibration profile '%s' not found, skipping calibration",
                self._config.calibration_profile,
            )
            return

        for step in hallucination_steps:
            for score_name in ("g_score", "u_score", "c_score", "faithfulness_score"):
                raw = getattr(step, score_name, 0.0)
                calibrated = profile.get_calibrated_score(score_name, raw)
                if calibrated is not None:
                    setattr(step, score_name, calibrated)
        logger.debug(
            "Applied calibration profile '%s' (kappa=%.3f) to %d steps",
            self._config.calibration_profile,
            profile.kappa_gsar,
            len(hallucination_steps),
        )

    def apply_faithfulness_check(
        self,
        hallucination_steps: List[HalluStepReport],
        trajectory: SecureTrajectory,
    ) -> None:
        from agent_trust_lab.hallukg.faithfulness import FaithfulnessChecker

        weights = self._config.anchor_type_weights
        checker = FaithfulnessChecker(nli_neutral_weight=self._config.nli_neutral_weight)
        disagreement_threshold = 0.3
        for step in hallucination_steps:
            if step.step_index >= len(trajectory.steps):
                continue
            traj_step = trajectory.steps[step.step_index]
            if traj_step.type in ("trap_injection", "action", "error"):
                continue
            evidence = step.evidence if step.evidence else ["no evidence"]
            gsar_score = step.faithfulness_score
            step.raw_gsar_faithfulness = gsar_score
            nli_score = checker.check([traj_step.content], evidence)
            alpha = weights.get(step.anchor_type, 0.5)
            blended = round(alpha * gsar_score + (1.0 - alpha) * nli_score, 4)
            disagreement = round(abs(gsar_score - nli_score), 4)

            step.nli_score = round(nli_score, 4)
            step.gsar_nli_disagreement = disagreement
            step.faithfulness_score = blended

            if disagreement >= disagreement_threshold:
                logger.warning(
                    "GSAR-NLI disagreement step %d: gsar=%.3f nli=%.3f disagreement=%.3f "
                    "(anchor=%s alpha=%.2f)",
                    step.step_index,
                    gsar_score,
                    nli_score,
                    disagreement,
                    step.anchor_type,
                    alpha,
                )

            logger.debug(
                "Faithfulness cross-check step %d: gsar=%.3f nli=%.3f alpha=%.3f (%s) "
                "blended=%.3f disagreement=%.3f",
                step.step_index,
                gsar_score,
                nli_score,
                alpha,
                step.anchor_type,
                blended,
                disagreement,
            )

    @staticmethod
    def _determine_anchor_type(triple: dict) -> str:
        """Determine anchor type from a single triple's evidence."""
        if triple.get("multi_hop"):
            return "multi_hop"
        evidence = triple.get("evidence", [])
        evidence_str = " ".join(str(e) for e in evidence)
        if "Semantic match" in evidence_str or "semantic" in evidence_str.lower():
            return "semantic"
        if "Token match" in evidence_str or "token" in evidence_str.lower():
            return "token_overlap"
        return "none"

    @staticmethod
    def _step_anchor_type(step_triples: list[dict]) -> str:
        """Determine dominant anchor type for a step from its triples."""
        if not step_triples:
            return "none"
        types = [HalluKGPipeline._determine_anchor_type(t) for t in step_triples]
        for preferred in ("semantic", "multi_hop", "token_overlap"):
            if preferred in types:
                return preferred
        return "none"

    @staticmethod
    def _merge_anchor_results(
        single_hop: list[dict[str, object]],
        multi_hop: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if len(single_hop) != len(multi_hop):
            return single_hop
        merged: list[dict[str, object]] = []
        for s, m in zip(single_hop, multi_hop):
            m_score_raw = m.get("anchor_score", 0.0)
            s_score_raw = s.get("anchor_score", 0.0)
            m_score = float(m_score_raw) if m_score_raw else 0.0  # type: ignore[arg-type]
            s_score = float(s_score_raw) if s_score_raw else 0.0  # type: ignore[arg-type]
            if m_score > s_score:
                merged.append(m)
            elif m.get("multi_hop") and m_score >= s_score * 0.8:
                m_evidence_raw = m.get("evidence", [])
                s_evidence_raw = s.get("evidence", [])
                m["evidence"] = (
                    list(s_evidence_raw) if isinstance(s_evidence_raw, list) else []
                ) + (list(m_evidence_raw) if isinstance(m_evidence_raw, list) else [])
                merged.append(m)
            else:
                merged.append(s)
        return merged
