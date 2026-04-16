"""
Report synthesis: generates a natural language audience research report
from aggregated simulation data.

Uses Ollama at moderate temperature to produce a structured report
that a film producer can act on.

Usage:
    python -m aggregation.synthesize --aggregation aggregation/reports/anora.json --model mistral
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from simulation.inference_config import (
    OLLAMA_CHAT_ENDPOINT,
    SYNTHESIS_CONFIG,
    config_to_ollama_options,
)

log = logging.getLogger("aggregation.synthesize")

SYNTHESIS_PROMPT = """\
You are a senior audience research analyst.

Results from a synthetic audience simulation of {n} agents reacting to a film:

{aggregation_json}

Write a concise, specific, actionable audience research report:

HEADLINE FINDING (1 sentence — the single most important thing a filmmaker needs to know)

OVERALL RECEPTION (2-3 sentences)

SEGMENT BREAKDOWN (one paragraph per segment — who loves it and why, who doesn't and why)

KEY FRICTION POINTS (name specific script elements, which segments, why)

KEY RESONANCE POINTS (specific elements driving positive reactions)

PREDICTED AUDIENCE PROFILE (concrete demographic terms)

ACTIONABLE RECOMMENDATION (one specific change improving reception in weakest segment without alienating strongest)

Be specific. Reference actual script elements. This will be read by a film producer making real decisions."""


async def synthesize_report(
    aggregated: dict,
    model: str = "mistral",
) -> str:
    """Generate a natural language report from aggregated simulation data."""
    n = aggregated.get("overall", {}).get("n", 0)
    prompt = SYNTHESIS_PROMPT.format(
        n=n,
        aggregation_json=json.dumps(aggregated, indent=2),
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": config_to_ollama_options(SYNTHESIS_CONFIG),
    }

    async with httpx.AsyncClient() as client:
        log.info("Generating synthesis report with model '%s'...", model)
        resp = await client.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


def main() -> None:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Synthesize audience research report")
    parser.add_argument("--aggregation", required=True,
                        help="Path to aggregation JSON")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--output", help="Path to save report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.aggregation) as f:
        aggregated = json.load(f)

    report = asyncio.run(synthesize_report(aggregated, model=args.model))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Report saved to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
