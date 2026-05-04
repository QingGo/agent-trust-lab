from unittest import mock

from agent_trust_lab.llm import (
    _DEFAULT_BASE_URL,
    create_langchain_llm,
    create_openai_client,
    get_api_key,
    get_base_url,
)
from agent_trust_lab.llm import (
    test_connection as _test_connection,
)


class TestDefaultBaseUrl:
    def test_constant_value(self):
        assert _DEFAULT_BASE_URL == "https://api.deepseek.com"


class TestGetApiKey:
    def test_explicit_arg_returns(self):
        assert get_api_key("sk-explicit") == "sk-explicit"

    def test_explicit_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        assert get_api_key("sk-explicit") == "sk-explicit"

    def test_deepseek_env_when_no_arg(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        assert get_api_key() == "sk-deepseek"

    def test_openai_fallback_when_no_deepseek(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        assert get_api_key() == "sk-openai"

    def test_deepseek_priority_over_openai(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        assert get_api_key() == "sk-deepseek"

    def test_empty_explicit_arg_uses_env(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        assert get_api_key("") == "sk-deepseek"

    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert get_api_key() is None


class TestGetBaseUrl:
    def test_explicit_arg_returns(self):
        assert get_base_url("https://custom.api.com") == "https://custom.api.com"

    def test_env_var_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.api.com")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.api.com")
        assert get_base_url() == "https://env.api.com"

    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        assert get_base_url() == _DEFAULT_BASE_URL

    def test_empty_explicit_uses_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.api.com")
        assert get_base_url("") == "https://env.api.com"


class TestCreateOpenaiClient:
    def test_creates_client_with_resolved_params(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("agent_trust_lab.llm.OpenAI") as mock_openai:
            mock_client = mock.MagicMock()
            mock_openai.return_value = mock_client

            client = create_openai_client(api_key="sk-override", base_url="https://custom.com")

            assert client is mock_client
            mock_openai.assert_called_once_with(
                api_key="sk-override", base_url="https://custom.com"
            )

    def test_uses_env_when_no_explicit_args(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("agent_trust_lab.llm.OpenAI") as mock_openai:
            create_openai_client()
            mock_openai.assert_called_once_with(
                api_key="sk-env", base_url=_DEFAULT_BASE_URL
            )

    def test_passes_empty_key_when_no_key_available(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("agent_trust_lab.llm.OpenAI") as mock_openai:
            create_openai_client()
            mock_openai.assert_called_once_with(
                api_key="", base_url=_DEFAULT_BASE_URL
            )


class TestCreateLangchainLlm:
    def test_creates_chat_openai_with_resolved_params(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("langchain_openai.ChatOpenAI") as mock_chat:
            create_langchain_llm(
                model="deepseek-test",
                api_key="sk-custom",
                base_url="https://custom.api",
                temperature=0.5,
            )
            mock_chat.assert_called_once_with(
                model="deepseek-test",
                api_key="sk-custom",
                base_url="https://custom.api",
                temperature=0.5,
            )

    def test_uses_default_model(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("langchain_openai.ChatOpenAI") as mock_chat:
            create_langchain_llm()
            call_args = mock_chat.call_args
            assert call_args[1]["model"] == "deepseek-v4-flash"
            assert call_args[1]["temperature"] == 0.0

    def test_uses_env_when_no_explicit_args(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

        with mock.patch("langchain_openai.ChatOpenAI") as mock_chat:
            create_langchain_llm()
            assert mock_chat.call_args[1]["api_key"] == "sk-env"
            assert mock_chat.call_args[1]["base_url"] == _DEFAULT_BASE_URL


class TestTestConnection:
    def test_no_api_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        success, msg = _test_connection()
        assert success is False
        assert "No API key" in msg

    def test_api_success_returns_true(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ):
            success, msg = _test_connection()
            assert success is True
            assert "Hello" in msg

    def test_api_error_returns_false(self, monkeypatch):
        from openai import APIError

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_request = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            "Rate limited", request=mock_request, body=None
        )

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ):
            success, msg = _test_connection()
            assert success is False
            assert "Rate limited" in msg

    def test_connection_error_returns_false(self, monkeypatch):
        from openai import APIConnectionError

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=mock.MagicMock()
        )

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ):
            success, msg = _test_connection()
            assert success is False

    def test_timeout_error_returns_false(self, monkeypatch):
        from openai import APITimeoutError

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=mock.MagicMock()
        )

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ):
            success, msg = _test_connection()
            assert success is False

    def test_unexpected_exception_returns_false(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("unexpected")

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ):
            success, msg = _test_connection()
            assert success is False
            assert "unexpected" in msg

    def test_passes_args_through(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = "Hi"
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch(
            "agent_trust_lab.llm.create_openai_client", return_value=mock_client
        ) as mock_create:
            _test_connection(model="custom-model", base_url="https://custom.com")
            mock_create.assert_called_once_with(
                model="custom-model", api_key="", base_url="https://custom.com"
            )
