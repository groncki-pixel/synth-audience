"""Provider tests.

Real network calls are mocked. The Gemini and Claude SDKs are stubbed via
``sys.modules`` injection so the tests run even when those packages are
not installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from simulation.inference_config import AGENT_CONFIG
from simulation.providers import OllamaProvider


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def test_ollama_provider_instantiates() -> None:
    p = OllamaProvider(model="mistral")
    assert p.name == "ollama"
    assert p.model == "mistral"
    assert p.last_input_tokens == 0
    assert p.last_output_tokens == 0


async def test_ollama_provider_generate_posts_correct_payload() -> None:
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "message": {"content": "hello"},
        "prompt_eval_count": 12,
        "eval_count": 5,
    }
    fake_response.raise_for_status = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_response)

    p = OllamaProvider(model="mistral", client=fake_client)
    result = await p.generate(
        system_prompt="sys",
        user_prompt="usr",
        config=AGENT_CONFIG,
    )

    assert result == "hello"
    assert p.last_input_tokens == 12
    assert p.last_output_tokens == 5

    fake_client.post.assert_awaited_once()
    call = fake_client.post.await_args
    url = call.args[0]
    payload = call.kwargs["json"]
    assert "/api/chat" in url
    assert payload["model"] == "mistral"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "usr"
    assert payload["options"]["temperature"] == AGENT_CONFIG.temperature


# ---------------------------------------------------------------------------
# Gemini — stub the SDK in sys.modules so import succeeds.
# ---------------------------------------------------------------------------

def _install_fake_genai(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Install a fake google.generativeai module and return (module, model)."""
    fake_model = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "gemini reply"
    fake_response.usage_metadata = MagicMock(
        prompt_token_count=20, candidates_token_count=7
    )
    fake_model.generate_content_async = AsyncMock(return_value=fake_response)

    fake_genai = types.SimpleNamespace(
        configure=MagicMock(),
        GenerativeModel=MagicMock(return_value=fake_model),
    )
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    return fake_genai, fake_model


def test_gemini_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Block dotenv from re-populating the env from a real .env file
    monkeypatch.setattr(
        "simulation.providers.gemini_provider._load_dotenv_once", lambda: None
    )
    from simulation.providers.gemini_provider import GeminiProvider
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiProvider()


async def test_gemini_provider_generate_calls_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "simulation.providers.gemini_provider._load_dotenv_once", lambda: None
    )
    fake_genai, fake_model = _install_fake_genai(monkeypatch)

    from simulation.providers.gemini_provider import GeminiProvider
    p = GeminiProvider(model="gemini-2.5-flash")
    out = await p.generate("sys", "usr", AGENT_CONFIG)

    assert out == "gemini reply"
    assert p.last_input_tokens == 20
    assert p.last_output_tokens == 7
    fake_genai.configure.assert_called_with(api_key="fake-key")
    fake_genai.GenerativeModel.assert_called_with(
        model_name="gemini-2.5-flash", system_instruction="sys"
    )
    fake_model.generate_content_async.assert_awaited_once()
    kwargs = fake_model.generate_content_async.await_args.kwargs
    assert kwargs["generation_config"]["temperature"] == AGENT_CONFIG.temperature
    assert kwargs["generation_config"]["max_output_tokens"] == AGENT_CONFIG.num_predict


# ---------------------------------------------------------------------------
# Claude — same pattern.
# ---------------------------------------------------------------------------

def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake_block = MagicMock()
    fake_block.text = "claude reply"
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_response.usage = MagicMock(input_tokens=22, output_tokens=9)

    fake_messages = MagicMock()
    fake_messages.create = AsyncMock(return_value=fake_response)
    fake_client = MagicMock()
    fake_client.messages = fake_messages

    fake_anthropic = types.SimpleNamespace(
        AsyncAnthropic=MagicMock(return_value=fake_client),
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return fake_anthropic, fake_client


def test_claude_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "simulation.providers.claude_provider._load_dotenv_once", lambda: None
    )
    from simulation.providers.claude_provider import ClaudeProvider
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider()


async def test_claude_provider_generate_calls_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(
        "simulation.providers.claude_provider._load_dotenv_once", lambda: None
    )
    fake_anthropic, fake_client = _install_fake_anthropic(monkeypatch)

    from simulation.providers.claude_provider import ClaudeProvider
    p = ClaudeProvider(model="claude-haiku-4-5")
    out = await p.generate("sys", "usr", AGENT_CONFIG)

    assert out == "claude reply"
    assert p.last_input_tokens == 22
    assert p.last_output_tokens == 9
    fake_anthropic.AsyncAnthropic.assert_called_with(api_key="fake-key")
    fake_client.messages.create.assert_awaited_once()
    kwargs = fake_client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "usr"}]
    assert kwargs["max_tokens"] == AGENT_CONFIG.num_predict
    assert kwargs["temperature"] == AGENT_CONFIG.temperature
