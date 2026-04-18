"""
Per-run token and dollar cost accounting.

A single ``CostTracker`` accumulates calls across an entire population run.
Per-call usage is recorded by the runner immediately after a successful
provider response (the providers expose ``last_input_tokens`` and
``last_output_tokens`` on the instance).

Prices are USD per 1M tokens. Wildcard models (``"*"``) match any model
under that provider, so Ollama (and any future self-hosted provider) gets
zero cost without needing per-model entries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Prices in USD per 1M tokens.
PRICE_TABLE: dict[tuple[str, str], dict[str, float]] = {
    ("gemini", "gemini-2.5-flash"): {"input": 0.0, "output": 0.0},
    ("claude", "claude-haiku-4-5"): {"input": 1.0, "output": 5.0},
    ("claude", "claude-sonnet-4-6"): {"input": 3.0, "output": 15.0},
    ("ollama", "*"): {"input": 0.0, "output": 0.0},
}


def _lookup_price(provider: str, model: str) -> dict[str, float]:
    if (provider, model) in PRICE_TABLE:
        return PRICE_TABLE[(provider, model)]
    if (provider, "*") in PRICE_TABLE:
        return PRICE_TABLE[(provider, "*")]
    return {"input": 0.0, "output": 0.0}


class CostTracker:
    """Accumulates token counts and USD cost for a single run."""

    def __init__(self) -> None:
        self._totals_in: int = 0
        self._totals_out: int = 0
        self._cost: float = 0.0
        self._calls: int = 0
        self._per_provider: dict[tuple[str, str], dict] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        )

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        price = _lookup_price(provider, model)
        cost = (
            (input_tokens / 1_000_000) * price["input"]
            + (output_tokens / 1_000_000) * price["output"]
        )

        self._totals_in += input_tokens
        self._totals_out += output_tokens
        self._cost += cost
        self._calls += 1

        bucket = self._per_provider[(provider, model)]
        bucket["calls"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["cost_usd"] += cost

    def summary(self) -> dict:
        return {
            "total_calls": self._calls,
            "total_input_tokens": self._totals_in,
            "total_output_tokens": self._totals_out,
            "total_cost_usd": round(self._cost, 6),
            "per_provider": {
                f"{prov}:{model}": {
                    "calls": data["calls"],
                    "input_tokens": data["input_tokens"],
                    "output_tokens": data["output_tokens"],
                    "cost_usd": round(data["cost_usd"], 6),
                }
                for (prov, model), data in self._per_provider.items()
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))
