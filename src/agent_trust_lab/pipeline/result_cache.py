"""Result cache for orchestrator pipeline.

Wraps agent_trust_lab.cache with cache-key computation from config.
"""

from typing import Optional

from agent_trust_lab.config import EvaluationConfig
from agent_trust_lab.log import get_logger
from agent_trust_lab.models.trap import EnhancedTrapDef

from agent_trust_lab.pipeline.models import EvaluationResult

logger = get_logger("orchestrator")


class ResultCache:
    """Handles cache get/put with config-driven key computation."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def get(self, trap: EnhancedTrapDef, runs_count: int = 1) -> Optional[EvaluationResult]:
        from agent_trust_lab.cache import (
            cache_get,
            cache_is_fresh,
            compute_cache_key,
        )

        cfg = self._config
        key = compute_cache_key(
            trap_id=trap.trap_id,
            model=cfg.model,
            judge_model=cfg.judge_model,
            tools=trap.tools,
            thinking_enabled=cfg.thinking_enabled,
            reasoning_effort=cfg.reasoning_effort,
            max_steps=cfg.max_steps,
            grounded_threshold=cfg.grounded_threshold,
            nli_neutral_weight=cfg.nli_neutral_weight,
            anchor_type_weights=cfg.anchor_type_weights,
            skip_extract_types=cfg.skip_extract_types,
            strict_mode=cfg.strict_mode,
            skip_hallukg=cfg.skip_hallukg,
            runs_count=runs_count,
            gsar_vote_enabled=cfg.gsar_vote_enabled,
            gsar_vote_models=cfg.gsar_vote_models,
        )
        if not cache_is_fresh(key, cfg.cache_ttl_days, cfg.cache_dir):
            return None
        data = cache_get(key, cfg.cache_dir)
        if data is None:
            return None
        try:
            return EvaluationResult.from_dict(data)
        except Exception as e:
            logger.warning("Failed to deserialize cached result for %s: %s", trap.trap_id, e)
            return None

    def put(
        self, trap: EnhancedTrapDef, result: EvaluationResult, runs_count: int = 1
    ) -> None:
        from agent_trust_lab.cache import cache_put, compute_cache_key

        cfg = self._config
        key = compute_cache_key(
            trap_id=trap.trap_id,
            model=cfg.model,
            judge_model=cfg.judge_model,
            tools=trap.tools,
            thinking_enabled=cfg.thinking_enabled,
            reasoning_effort=cfg.reasoning_effort,
            max_steps=cfg.max_steps,
            grounded_threshold=cfg.grounded_threshold,
            nli_neutral_weight=cfg.nli_neutral_weight,
            anchor_type_weights=cfg.anchor_type_weights,
            skip_extract_types=cfg.skip_extract_types,
            strict_mode=cfg.strict_mode,
            skip_hallukg=cfg.skip_hallukg,
            runs_count=runs_count,
            gsar_vote_enabled=cfg.gsar_vote_enabled,
            gsar_vote_models=cfg.gsar_vote_models,
        )
        try:
            cache_put(key, result.to_dict(), cfg.cache_dir)
        except Exception as e:
            logger.warning("Failed to cache result for %s: %s", trap.trap_id, e)
