"""
Gemini InferenceProvider.

Uses the official ``google-generativeai`` library. The API key is read from
the ``GEMINI_API_KEY`` environment variable; ``.env`` files are loaded
automatically via ``python-dotenv``. Rate-limit and other transient errors
are caught and reported as ``None`` so the runner's retry loop can take
over.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from simulation.context_guard import CHARS_PER_TOKEN
from simulation.inference_config import InferenceConfig

log = logging.getLogger("simulation.providers.gemini")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class GeminiProvider:
    """InferenceProvider backed by Google Gemini."""

    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        _load_dotenv_once()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a key at "
                "https://aistudio.google.com/app/apikey and add it to .env."
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=key)
        self._genai = genai
        self.model = model
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def _build_generation_config(self, config: InferenceConfig) -> dict:
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.num_predict,
        }

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: InferenceConfig,
        timeout_seconds: float = 120,
    ) -> Optional[str]:
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )
        try:
            resp = await model.generate_content_async(
                user_prompt,
                generation_config=self._build_generation_config(config),
                request_options={"timeout": timeout_seconds},
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "quota" in msg or "429" in msg:
                log.warning("Gemini rate-limited: %s", exc)
                return None
            log.warning("Gemini call failed: %s", exc)
            return None

        text = getattr(resp, "text", None) or ""

        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            self.last_input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            self.last_output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        else:
            log.warning("Gemini response missing usage_metadata; estimating from chars")
            self.last_input_tokens = max(
                1, (len(system_prompt) + len(user_prompt)) // CHARS_PER_TOKEN
            )
            self.last_output_tokens = max(1, len(text) // CHARS_PER_TOKEN)

        return text
