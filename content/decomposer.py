"""
Content decomposer: analyzes script structure and friction/resonance points.

One script → one LLM call at low temperature (analytical task, no persona).
Produces a structured JSON decomposition that agents react to alongside
the raw script text.

Can use either Ollama (local) or be adapted for other providers.

Usage:
    python -m content.decomposer --script content/scripts/anora_act1.txt --model mistral
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

from simulation.inference_config import (
    DECOMPOSITION_CONFIG,
    OLLAMA_CHAT_ENDPOINT,
    config_to_ollama_options,
)

log = logging.getLogger("content.decomposer")

DECOMPOSITION_PROMPT = """\
You are a content analyst specializing in narrative structure and audience reception.

Analyze this film script excerpt. Be specific — reference actual scenes, dialogue, and structural choices.

Return ONLY valid JSON matching this schema exactly:

{
  "narrative_structure": {
    "pacing_assessment": string,
    "act_structure": string,
    "key_beats": [string]
  },
  "character_frames": [
    {
      "character": string,
      "how_framed": string,
      "agency_level": "high" | "medium" | "low",
      "values_embodied": [string]
    }
  ],
  "thematic_claims": [
    {
      "theme": string,
      "how_expressed": string,
      "implicit_argument": string
    }
  ],
  "tonal_profile": {
    "primary_tone": string,
    "tone_shifts": [string],
    "genre_signals": [string]
  },
  "friction_points": [
    {
      "element": string,
      "description": string,
      "potential_positive_read": string,
      "potential_negative_read": string,
      "sensitive_demographics": [string]
    }
  ],
  "resonance_points": [
    {
      "element": string,
      "why_compelling": string,
      "target_demographics": [string]
    }
  ]
}"""


async def decompose_content(
    script_text: str,
    title: str,
    model: str = "mistral",
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Decompose a script into structured friction/resonance analysis.

    Args:
        script_text: The raw script content.
        title: Title slug used for saving the output file.
        model: Ollama model to use for decomposition.
        save_dir: Directory to save decomposition JSON (default: content/decompositions/).

    Returns:
        Parsed decomposition dict.
    """
    if save_dir is None:
        save_dir = Path("content/decompositions")
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{DECOMPOSITION_PROMPT}\n\nScript:\n\n{script_text}"},
        ],
        "stream": False,
        "options": config_to_ollama_options(DECOMPOSITION_CONFIG),
    }

    async with httpx.AsyncClient() as client:
        log.info("Decomposing content for '%s' with model '%s'...", title, model)
        resp = await client.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=180.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")

    # Parse — strip code fences if present
    text = raw.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else len(text)
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    decomposition = json.loads(text)

    out_path = save_dir / f"{title}.json"
    with open(out_path, "w") as f:
        json.dump(decomposition, f, indent=2)
    log.info("Decomposition saved to %s", out_path)

    return decomposition


def main() -> None:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Decompose script content")
    parser.add_argument("--script", required=True, help="Path to script text file")
    parser.add_argument("--title", help="Title slug (default: filename stem)")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    script_path = Path(args.script)
    script_text = script_path.read_text()
    title = args.title or script_path.stem

    result = asyncio.run(decompose_content(script_text, title, model=args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
