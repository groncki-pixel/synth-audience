"""
Ollama InferenceProvider.

Wraps the local Ollama /api/chat endpoint. Behavior is identical to the
original ``simulation.runner._call_ollama`` helper that this provider
replaces — same payload shape, same options mapping, same failure handling.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from simulation.context_guard import CHARS_PER_TOKEN
from simulation.inference_config import (
    InferenceConfig,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_ENDPOINT,
    config_to_ollama_options,
)

log = logging.getLogger("simulation.providers.ollama")


class OllamaProvider:
    """InferenceProvider implementation backed by a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = OLLAMA_BASE_URL,
        chat_endpoint: str = OLLAMA_CHAT_ENDPOINT,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.chat_endpoint = chat_endpoint
        self._client = client
        self._owns_client = client is None
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Verify Ollama is reachable."""
        client = await self._get_client()
        try:
            resp = await client.get(self.base_url, timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: InferenceConfig,
        timeout_seconds: float = 120,
    ) -> Optional[str]:
        client = await self._get_client()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": config_to_ollama_options(config),
        }

        try:
            resp = await client.post(
                self.chat_endpoint,
                json=payload,
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            log.warning("Ollama call timed out after %ss", timeout_seconds)
            return None
        except httpx.HTTPStatusError as exc:
            log.warning("Ollama HTTP error: %s", exc.response.status_code)
            return None
        except httpx.ConnectError:
            log.error("Cannot connect to Ollama at %s", self.base_url)
            return None

        content = data.get("message", {}).get("content", "")

        prompt_tokens = data.get("prompt_eval_count")
        eval_tokens = data.get("eval_count")
        if prompt_tokens is None:
            log.warning("Ollama response missing prompt_eval_count; estimating from chars")
            prompt_tokens = max(1, (len(system_prompt) + len(user_prompt)) // CHARS_PER_TOKEN)
        if eval_tokens is None:
            log.warning("Ollama response missing eval_count; estimating from chars")
            eval_tokens = max(1, len(content) // CHARS_PER_TOKEN)
        self.last_input_tokens = int(prompt_tokens)
        self.last_output_tokens = int(eval_tokens)

        return content
