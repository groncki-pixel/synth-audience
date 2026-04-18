"""
Provider protocol for inference backends.

Defines the minimal async interface every inference provider must implement.
Concrete providers (Ollama, Gemini, Claude) live alongside this file and are
selected at runtime by the simulation runner and the pipeline CLI.

Each provider also exposes the most recent call's token usage on the instance
attributes ``last_input_tokens`` and ``last_output_tokens`` so the cost tracker
can record per-call usage without tightening the protocol surface.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from simulation.inference_config import InferenceConfig


@runtime_checkable
class InferenceProvider(Protocol):
    """Minimal async provider interface used by the simulation runner."""

    last_input_tokens: int
    last_output_tokens: int
    name: str
    model: str

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: InferenceConfig,
        timeout_seconds: float = 120,
    ) -> Optional[str]:
        """Return raw model response text or None on failure."""
        ...
