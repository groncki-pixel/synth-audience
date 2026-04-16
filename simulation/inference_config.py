"""
Inference configuration for Ollama API calls.

Three task profiles with different temperature/sampling settings:
  - agent:         high temperature for persona variance, tight token limit
  - decomposition: low temperature for analytical consistency
  - synthesis:     moderate temperature for readable reports

Modeled on the OpenClaw extra-params pattern (extra-params.ts) where
different call types get different temperature/top_p/max_tokens settings
routed through the same provider interface.
"""

from __future__ import annotations

from dataclasses import dataclass

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"


@dataclass(frozen=True)
class InferenceConfig:
    """Parameters passed to Ollama /api/chat."""
    temperature: float
    top_p: float
    repeat_penalty: float
    num_predict: int          # max tokens to generate


# Agent simulation calls — high temp for persona variance, short output
AGENT_CONFIG = InferenceConfig(
    temperature=0.85,
    top_p=0.88,
    repeat_penalty=1.0,
    num_predict=600,
)

# Content decomposition — analytical, low variance
DECOMPOSITION_CONFIG = InferenceConfig(
    temperature=0.3,
    top_p=0.90,
    repeat_penalty=1.0,
    num_predict=1500,
)

# Report synthesis — moderate creativity
SYNTHESIS_CONFIG = InferenceConfig(
    temperature=0.5,
    top_p=0.92,
    repeat_penalty=1.0,
    num_predict=1200,
)


def config_to_ollama_options(config: InferenceConfig) -> dict:
    """Convert InferenceConfig to Ollama API 'options' dict."""
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repeat_penalty": config.repeat_penalty,
        "num_predict": config.num_predict,
    }
