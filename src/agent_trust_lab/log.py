import logging
import sys
from typing import Optional

ROOT_LOGGER_NAME = "agent_trust_lab"
_log_initialized = False


def setup_logging(
    level: int = logging.WARNING,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the agent-trust-lab package.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.INFO).
        log_file: Optional file path for log output. Logs to stderr if None.
    """
    global _log_initialized
    if _log_initialized:
        return

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.handlers.clear()

    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    _log_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name, scoped under agent_trust_lab."""
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def cli_verbosity_to_level(verbose: int) -> int:
    """Convert CLI verbosity count to a logging level.

    0 (default) -> WARNING, -v -> INFO, -vv -> DEBUG.
    """
    if verbose >= 2:
        return logging.DEBUG
    if verbose == 1:
        return logging.INFO
    return logging.WARNING
