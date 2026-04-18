"""Inference provider abstraction.

Exports the protocol and the three concrete providers shipped today.
"""

from simulation.providers.base import InferenceProvider
from simulation.providers.claude_provider import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeProvider,
)
from simulation.providers.gemini_provider import (
    DEFAULT_GEMINI_MODEL,
    GeminiProvider,
)
from simulation.providers.ollama_provider import OllamaProvider

__all__ = [
    "InferenceProvider",
    "OllamaProvider",
    "GeminiProvider",
    "ClaudeProvider",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_GEMINI_MODEL",
]
