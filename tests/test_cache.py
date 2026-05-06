import os
import tempfile

from agent_trust_lab.cache import (
    cache_clear,
    cache_get,
    cache_invalidate,
    cache_is_fresh,
    cache_put,
    compute_cache_key,
)


class TestComputeCacheKey:
    def test_deterministic(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        assert k1 == k2
        assert len(k1) == 64

    def test_different_trap_id(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key("trap_b", "m1", "", [{"name": "shell"}])
        assert k1 != k2

    def test_different_model(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key("trap_a", "m2", "", [{"name": "shell"}])
        assert k1 != k2

    def test_judge_model_affects_key(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key("trap_a", "m1", "judge-other", [{"name": "shell"}])
        assert k1 != k2

    def test_different_tools(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key("trap_a", "m1", "", [{"name": "file_read"}])
        assert k1 != k2

    def test_tools_order_independent(self):
        k1 = compute_cache_key("trap_a", "m1", "", [
            {"name": "shell"}, {"name": "file_read"}
        ])
        k2 = compute_cache_key("trap_a", "m1", "", [
            {"name": "file_read"}, {"name": "shell"}
        ])
        assert k1 == k2

    def test_differs_with_thinking(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key(
            "trap_a", "m1", "", [{"name": "shell"}], thinking_enabled=True
        )
        assert k1 != k2

    def test_differs_with_reasoning_effort(self):
        k1 = compute_cache_key(
            "trap_a", "m1", "", [{"name": "shell"}], thinking_enabled=True
        )
        k2 = compute_cache_key(
            "trap_a", "m1", "", [{"name": "shell"}],
            thinking_enabled=True, reasoning_effort="high",
        )
        assert k1 != k2

    def test_differs_with_skip_hallukg(self):
        k1 = compute_cache_key("trap_a", "m1", "", [{"name": "shell"}])
        k2 = compute_cache_key(
            "trap_a", "m1", "", [{"name": "shell"}], skip_hallukg=True
        )
        assert k1 != k2


class TestCacheGetPut:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_put_and_get(self):
        key = "test_abc123"
        data = {"trap_id": "test", "score": 0.95}
        cache_put(key, data, self.tmp)
        result = cache_get(key, self.tmp)
        assert result is not None
        assert result["trap_id"] == "test"
        assert result["score"] == 0.95

    def test_get_missing_returns_none(self):
        result = cache_get("nonexistent", self.tmp)
        assert result is None

    def test_put_overwrites(self):
        key = "overwrite_test"
        cache_put(key, {"v": 1}, self.tmp)
        cache_put(key, {"v": 2}, self.tmp)
        result = cache_get(key, self.tmp)
        assert result["v"] == 2

    def test_get_corrupted_file(self):
        key = "corrupted"
        path = os.path.join(self.tmp, f"{key}.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        result = cache_get(key, self.tmp)
        assert result is None


class TestCacheFreshness:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_fresh_within_ttl(self):
        key = "fresh_test"
        cache_put(key, {"a": 1}, self.tmp)
        fresh = cache_is_fresh(key, ttl_days=7, cache_dir=self.tmp)
        assert fresh is True

    def test_not_fresh_with_zero_ttl(self):
        key = "stale_test"
        cache_put(key, {"a": 1}, self.tmp)
        fresh = cache_is_fresh(key, ttl_days=0, cache_dir=self.tmp)
        assert fresh is False

    def test_missing_file_not_fresh(self):
        fresh = cache_is_fresh("nonexistent", ttl_days=7, cache_dir=self.tmp)
        assert fresh is False


class TestCacheInvalidate:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_invalidate_existing(self):
        key = "inv_test"
        cache_put(key, {"a": 1}, self.tmp)
        removed = cache_invalidate(key, self.tmp)
        assert removed is True
        assert cache_get(key, self.tmp) is None

    def test_invalidate_missing(self):
        removed = cache_invalidate("nonexistent", self.tmp)
        assert removed is False


class TestCacheClear:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_clear_all(self):
        cache_put("a", {"v": 1}, self.tmp)
        cache_put("b", {"v": 2}, self.tmp)
        count = cache_clear(cache_dir=self.tmp)
        assert count == 2
        assert cache_get("a", self.tmp) is None

    def test_clear_empty_dir(self):
        count = cache_clear(cache_dir=self.tmp)
        assert count == 0

    def test_clear_nonexistent_dir(self):
        count = cache_clear(cache_dir="/nonexistent/dir/path/xyz")
        assert count == 0
