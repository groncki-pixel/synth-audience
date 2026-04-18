"""
Claude (Anthropic) InferenceProvider.

Uses the official ``anthropic`` async client. The API key is read from
``ANTHROPIC_API_KEY``; ``.env`` files are loaded automatically when
``python-dotenv`` is installed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from simulation.context_guard import CHARS_PER_TOKEN
from simulation.inference_config import InferenceConfig

log = logging.getLogger("simulation.providers.claude")

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class ClaudeProvider:
    """InferenceProvider backed by the Anthropic Claude API."""

    name = "claude"

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        _load_dotenv_once()
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Create a key at "
                "https://console.anthropic.com/ and add it to .env."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic is not installed. Run: pip install anthropic"
            ) from exc

        self._client = anthropic.AsyncAnthropic(api_key=key)
        self.model = model
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: InferenceConfig,
        timeout_seconds: float = 120,
    ) -> Optional[str]:
        try:
            resp = await self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=config.num_predict,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg or "overloaded" in msg:
                log.warning("Claude rate-limited: %s", exc)
                return None
            log.warning("Claude call failed: %s", exc)
            return None

        text_parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
        text = "".join(text_parts)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            self.last_output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        else:
            log.warning("Claude response missing usage; estimating from chars")
            self.last_input_tokens = max(
                1, (len(system_prompt) + len(user_prompt)) // CHARS_PER_TOKEN
            )
            self.last_output_tokens = max(1, len(text) // CHARS_PER_TOKEN)

        return text
