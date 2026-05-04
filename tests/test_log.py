import logging
import os
import tempfile

from agent_trust_lab.log import (
    ROOT_LOGGER_NAME,
    cli_verbosity_to_level,
    get_logger,
    setup_logging,
)


class TestRootLoggerName:
    def test_constant_value(self):
        assert ROOT_LOGGER_NAME == "agent_trust_lab"


class TestCliVerbosityToLevel:
    def test_default_warning(self):
        assert cli_verbosity_to_level(0) == logging.WARNING

    def test_single_verbose_info(self):
        assert cli_verbosity_to_level(1) == logging.INFO

    def test_double_verbose_debug(self):
        assert cli_verbosity_to_level(2) == logging.DEBUG

    def test_triple_verbose_debug(self):
        assert cli_verbosity_to_level(3) == logging.DEBUG

    def test_negative_verbose_warning(self):
        assert cli_verbosity_to_level(-1) == logging.WARNING


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_scopes_name_under_root(self):
        logger = get_logger("test_module")
        assert logger.name == "agent_trust_lab.test_module"

    def test_already_scoped_name_not_double_prefixed(self):
        logger = get_logger("agent_trust_lab.test_module")
        assert logger.name == "agent_trust_lab.test_module"

    def test_empty_name(self):
        logger = get_logger("")
        assert logger.name == "agent_trust_lab."


class TestSetupLogging:
    def teardown_method(self):
        import agent_trust_lab.log as log_mod

        log_mod._log_initialized = False
        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.handlers.clear()

    def test_defaults_writes_to_stderr(self):
        setup_logging()
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

    def test_custom_level(self):
        setup_logging(level=logging.DEBUG)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        assert root.level == logging.DEBUG

    def test_log_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            setup_logging(level=logging.INFO, log_file=log_path)
            logger = get_logger("test_module")
            logger.warning("test message")

            handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
            for h in handlers:
                h.flush()
                if hasattr(h, "close"):
                    h.close()

            assert os.path.exists(log_path)
            with open(log_path, "r") as f:
                content = f.read()
            assert "test message" in content

    def test_idempotent_second_call_noop(self):
        setup_logging(level=logging.INFO)
        root = logging.getLogger(ROOT_LOGGER_NAME)
        handler_count = len(root.handlers)
        first_level = root.level

        setup_logging(level=logging.DEBUG)
        assert len(root.handlers) == handler_count
        assert root.level == first_level
