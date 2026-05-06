import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from agent_trust_lab.config import RESULT_CACHE_DIR
from agent_trust_lab.log import get_logger

logger = get_logger("cache")


def compute_cache_key(
    trap_id: str,
    model: str,
    judge_model: str,
    tools: List[Dict[str, Any]],
    thinking_enabled: bool = False,
    reasoning_effort: str = "",
    max_steps: int = 10,
    grounded_threshold: float = 0.3,
    nli_neutral_weight: float = 0.5,
    anchor_type_weights: Optional[Dict[str, float]] = None,
    skip_extract_types: Optional[List[str]] = None,
    strict_mode: bool = False,
    skip_hallukg: bool = False,
) -> str:
    """Compute a deterministic cache key from evaluation parameters.

    Excludes sensitive fields (api_key, base_url, docker_host) and
    file paths (trap_library_path, codebase_path, test_suite_path).

    Returns a 64-char hex string (SHA-256).
    """
    weights = dict(anchor_type_weights) if anchor_type_weights else {}
    skip_types = sorted(skip_extract_types) if skip_extract_types else []
    tools_serialized = json.dumps(
        sorted(tools, key=lambda x: json.dumps(x, sort_keys=True)),
        sort_keys=True,
    )

    components = [
        trap_id,
        model,
        judge_model or model,
        "1" if thinking_enabled else "0",
        reasoning_effort,
        str(max_steps),
        f"{grounded_threshold:.6f}",
        f"{nli_neutral_weight:.6f}",
        json.dumps(weights, sort_keys=True),
        json.dumps(skip_types),
        "1" if strict_mode else "0",
        "1" if skip_hallukg else "0",
        tools_serialized,
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_dir(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cache_path(key: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def cache_get(key: str, cache_dir: str = RESULT_CACHE_DIR) -> Optional[Dict[str, Any]]:
    """Retrieve a cached result dict by key.

    Returns None if the cache file does not exist or is corrupted.
    """
    path = _cache_path(key, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("data")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read cache file %s: %s", path, e)
        return None


def cache_put(key: str, data: Dict[str, Any], cache_dir: str = RESULT_CACHE_DIR) -> None:
    """Store a result dict in the cache."""
    _ensure_dir(cache_dir)
    path = _cache_path(key, cache_dir)
    payload = {"key": key, "cached_at": time.time(), "data": data}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.debug("Cached result: %s", key)
    except OSError as e:
        logger.warning("Failed to write cache file %s: %s", path, e)


def cache_invalidate(key: str, cache_dir: str = RESULT_CACHE_DIR) -> bool:
    """Remove a cached result by key. Returns True if removed successfully."""
    path = _cache_path(key, cache_dir)
    if os.path.isfile(path):
        try:
            os.remove(path)
            return True
        except OSError as e:
            logger.warning("Failed to remove cache file %s: %s", path, e)
    return False


def cache_is_fresh(key: str, ttl_days: int, cache_dir: str = RESULT_CACHE_DIR) -> bool:
    """Check whether a cached result is still within its TTL.

    Returns True if the cache exists and is under ttl_days old.
    Returns False if the cache does not exist, is corrupted, or is expired.
    """
    path = _cache_path(key, cache_dir)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cached_at = payload.get("cached_at", 0)
        age_seconds = time.time() - cached_at
        max_age = ttl_days * 86400
        return age_seconds < max_age
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning("Failed to check cache freshness for %s: %s", key, e)
        return False


def cache_clear(
    older_than_days: Optional[int] = None,
    cache_dir: str = RESULT_CACHE_DIR,
) -> int:
    """Remove all cached results, optionally only those older than a threshold.

    Returns the number of cache entries removed.
    """
    if not os.path.isdir(cache_dir):
        return 0
    count = 0
    max_age = (older_than_days or 0) * 86400 if older_than_days else None
    now = time.time()

    for fname in os.listdir(cache_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(cache_dir, fname)
        if max_age is not None:
            try:
                mtime = os.path.getmtime(path)
                if now - mtime < max_age:
                    continue
            except OSError:
                pass
        try:
            os.remove(path)
            count += 1
        except OSError:
            pass

    if count > 0:
        logger.info("Cleared %d cached result(s) from %s", count, cache_dir)
    return count
