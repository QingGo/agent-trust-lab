.PHONY: help lint format test-unit test-all test-manager test-models test-mutator smoke clean install test-integration test-docker test-slow test-e2e install-onnx

# Default target
help:
	@echo "agent-trust-lab dev targets"
	@echo ""
	@echo "  make lint            ruff check (L0, <0.2s)"
	@echo "  make format          ruff format"
	@echo "  make test-unit       all tests except integration/docker/slow/e2e (L1)"
	@echo "  make test-all        full suite incl. YAML I/O (~180s)"
	@echo "  make test-manager    trap manager tests only"
	@echo "  make test-models     model validation tests only"
	@echo "  make test-mutator    mutator tests only"
	@echo "  make test-integration  integration tests (needs API key)"
	@echo "  make test-docker     Docker integration tests"
	@echo "  make test-slow       ONNX integration tests"
	@echo "  make test-e2e        full E2E tests"
	@echo "  make smoke           validate-traps + log/llm tests (L4)"
	@echo "  make clean           remove cache files"
	@echo "  make install         editable install with Tsinghua mirror"
	@echo "  make install-onnx    install ONNX deps (pip, not uv)"

VENV = .venv
PYTHON = $(VENV)/bin/python
RUFF = $(VENV)/bin/ruff
MIRROR = https://pypi.tuna.tsinghua.edu.cn/simple

$(VENV)/bin/activate:
	uv venv

install: $(VENV)/bin/activate
	uv pip install -e . --index-url $(MIRROR)
	uv pip install --group dev --index-url $(MIRROR)

# L0: static check (<0.2s)
lint:
	$(RUFF) check src/ tests/

# L0: type check
typecheck:
	basedpyright src/

# L0: auto-format
format:
	$(RUFF) format src/ tests/

# L1: all non-integration tests, skips slow/expensive suites
test-unit:
	$(PYTHON) -m pytest tests/ -v -m "not integration and not docker and not slow and not e2e"

# L1: individual test files
test-models:
	$(PYTHON) -m pytest tests/test_models.py -v

test-mutator:
	$(PYTHON) -m pytest tests/test_trap_mutator.py -v

test-manager:
	$(PYTHON) -m pytest tests/test_trap_manager.py -v

# L1: full test suite (~180s)
test-all:
	$(PYTHON) -m pytest tests/ -v

# L4: smoke test — validate entire trap library loads correctly + fast unit tests
smoke:
	$(PYTHON) -m agent_trust_lab.cli validate-traps
	$(PYTHON) -m pytest tests/test_log.py tests/test_llm.py -q

# L3: integration tests — requires DEEPSEEK_API_KEY (auto-skipped otherwise)
test-integration:
	$(PYTHON) -m pytest tests/integration/ -v -m "integration"

# L3: Docker integration tests (auto-skipped when Docker unavailable)
test-docker:
	$(PYTHON) -m pytest tests/integration/ -v -m "docker"

# L3: ONNX integration tests (auto-skipped when models not cached)
test-slow:
	$(PYTHON) -m pytest tests/integration/ -v -m "slow"

# L4: full E2E tests (needs API key + Docker + ONNX)
test-e2e:
	$(PYTHON) -m pytest tests/integration/ -v -m "e2e"

# DevEx: remove cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true

# DevEx: install ONNX dependencies (use pip, not uv — uv<0.9.19 macOS tag bug)
install-onnx:
	$(PYTHON) -m pip install onnxruntime tokenizers --index-url $(MIRROR)
