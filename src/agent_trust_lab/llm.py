import os
import time
from typing import Callable, Optional, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from agent_trust_lab.config import DEFAULT_MODEL
from agent_trust_lab.log import get_logger

load_dotenv()

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_MAX_RETRIES = 3
logger = get_logger("llm")

_T = TypeVar("_T")
_RETRYABLE_ERRORS = (APIError, APIConnectionError, APITimeoutError, RateLimitError)


def get_api_key(api_key: str = "") -> Optional[str]:
    """Resolve API key: explicit arg > DEEPSEEK_API_KEY env > OPENAI_API_KEY env."""
    if api_key:
        return api_key
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        return deepseek_key
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        return openai_key
    return None


def get_base_url(base_url: str = "") -> str:
    """Resolve base URL: explicit arg > DEEPSEEK_BASE_URL env > default DeepSeek."""
    if base_url:
        return base_url
    env_url = os.environ.get("DEEPSEEK_BASE_URL", "")
    if env_url:
        return env_url
    return _DEFAULT_BASE_URL


def create_openai_client(
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    max_retries: int = _MAX_RETRIES,
) -> OpenAI:
    """Create an OpenAI-compatible client. All params are explicit for future multi-model use."""
    resolved_key = get_api_key(api_key)
    resolved_url = get_base_url(base_url)
    return OpenAI(api_key=resolved_key or "", base_url=resolved_url, max_retries=max_retries)


def create_langchain_llm(
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    max_retries: int = _MAX_RETRIES,
):
    """Create a LangChain ChatOpenAI instance configured for the provider."""
    from langchain_openai import ChatOpenAI

    resolved_key = get_api_key(api_key)
    resolved_url = get_base_url(base_url)
    return ChatOpenAI(
        model=model,
        api_key=resolved_key or "",  # pyright: ignore[reportArgumentType]
        base_url=resolved_url,
        temperature=temperature,
        max_retries=max_retries,
    )


def test_connection(
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    base_url: str = "",
) -> tuple[bool, str]:
    """Verify API connectivity. Returns (success, message)."""
    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return False, "No API key found (set DEEPSEEK_API_KEY in .env)"
    try:
        client = create_openai_client(model=model, api_key=api_key, base_url=base_url)
        response: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        content = response.choices[0].message.content or ""
        return True, f"OK: {content[:100]}"
    except _RETRYABLE_ERRORS as e:
        logger.warning("API connection test failed: %s", e)
        return False, str(e)
    except Exception as e:
        logger.error("Unexpected error during connection test: %s", e, exc_info=True)
        return False, str(e)


def retry_with_backoff(
    func: Callable[[], _T],
    max_retries: int = _MAX_RETRIES,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> _T:
    """Call func with exponential backoff on transient API errors.

    Raises the last exception after exhausting retries.
    Returns the result of func on success.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return func()
        except _RETRYABLE_ERRORS as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "Retry %d/%d after %.1fs: %s", attempt + 1, max_retries, delay, e
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
