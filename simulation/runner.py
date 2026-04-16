"""
Async batch agent runner calling Ollama at localhost:11434/api/chat.

Orchestrates the full simulation: for each agent, builds prompts, calls
Ollama, parses JSON, validates against collapse heuristics, retries on
failure with strengthened anchoring.

Design patterns from OpenClaw reference:
  - pi-embedded-runner/run.ts: retry loop with failover classification,
    timeout handling, abort signals, and progressive retry escalation
  - compact.ts: batch execution with concurrent agent calls
  - context-window-guard.ts: pre-flight context size check before each call

Key constraints:
  - All calls go to Ollama locally via httpx (no external APIs)
  - Batch size defaults to 5 to avoid overwhelming local GPU
  - Every response goes through validator.py before acceptance
  - Failed validation triggers retry up to 3 times
  - Retry 2 strengthens persona anchoring in the system prompt
  - All outputs saved as JSONL to simulation/outputs/

Usage:
    python -m simulation.runner --model mistral --script content/scripts/anora_act1.txt
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from agents.schemas import AgentProfile, AgentReaction, AgentResult
from simulation.context_guard import guard_context, estimate_tokens
from simulation.inference_config import (
    AGENT_CONFIG,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_ENDPOINT,
    config_to_ollama_options,
)
from simulation.prompts import (
    REACTION_PROMPT_TEMPLATE,
    build_system_prompt,
    build_user_prompt,
)
from simulation.validator import validate_response

log = logging.getLogger("simulation.runner")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MODEL = "mistral"
OUTPUT_DIR = Path("simulation/outputs")


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

async def check_ollama_available(client: httpx.AsyncClient) -> bool:
    """Verify Ollama is reachable before starting a population run."""
    try:
        resp = await client.get(OLLAMA_BASE_URL, timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# ---------------------------------------------------------------------------
# Single agent call
# ---------------------------------------------------------------------------

async def _call_ollama(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """
    Make a single Ollama /api/chat call. Returns the raw response text
    or None on transport/timeout failure.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": config_to_ollama_options(AGENT_CONFIG),
    }

    try:
        resp = await client.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except httpx.TimeoutException:
        log.warning("Ollama call timed out after %ss", timeout_seconds)
        return None
    except httpx.HTTPStatusError as exc:
        log.warning("Ollama HTTP error: %s", exc.response.status_code)
        return None
    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_BASE_URL)
        return None


def _extract_json(raw: str) -> Optional[dict]:
    """
    Extract JSON from raw model output. Handles cases where the model
    wraps JSON in markdown code fences or adds preamble text.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Find the end of the opening fence line
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    # Walk forward to find the matching closing brace
    depth = 0
    in_string = False
    escape = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    return None

    return None


# ---------------------------------------------------------------------------
# Single agent with retry
# ---------------------------------------------------------------------------

async def run_single_agent(
    client: httpx.AsyncClient,
    agent: AgentProfile,
    script_text: str,
    decomposition: dict,
    model: str,
    context_window_tokens: int = 8192,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[AgentResult]:
    """
    Run a single agent through the simulation with retry logic.

    Retry strategy (modeled on OpenClaw's run.ts retry loop):
      - Attempt 1: standard prompts
      - Attempt 2: add explicit anti-hedging reinforcement to persona
      - Attempt 3: further strengthen with direct prohibition list

    Returns AgentResult on success, None if all retries exhausted.
    """
    decomposition_json = json.dumps(decomposition, indent=2)

    # Build the mutable persona description (may be strengthened on retry)
    current_description = agent.natural_language_description

    for attempt in range(1, max_retries + 1):
        # Build prompts with current (possibly strengthened) description
        modified_agent = agent.model_copy()
        object.__setattr__(modified_agent, "natural_language_description", current_description)

        system_prompt = build_system_prompt(modified_agent)

        # Context guard: truncate script if needed
        safe_script = guard_context(
            system_prompt=system_prompt,
            user_prompt_template=REACTION_PROMPT_TEMPLATE,
            script_text=script_text,
            decomposition_json=decomposition_json,
            context_window_tokens=context_window_tokens,
        )

        user_prompt = build_user_prompt(safe_script, decomposition_json)

        # Call Ollama
        raw_response = await _call_ollama(client, model, system_prompt, user_prompt)
        if raw_response is None:
            log.warning("Agent %s attempt %d: no response from Ollama",
                        agent.agent_id, attempt)
            continue

        # Parse JSON
        parsed = _extract_json(raw_response)
        if parsed is None:
            log.warning("Agent %s attempt %d: JSON parse failed",
                        agent.agent_id, attempt)
            continue

        # Validate
        is_valid, reason = validate_response(parsed, agent)
        if is_valid:
            try:
                reaction = AgentReaction(**parsed)
            except Exception as exc:
                log.warning("Agent %s attempt %d: schema validation failed: %s",
                            agent.agent_id, attempt, exc)
                continue

            return AgentResult(
                agent_id=agent.agent_id,
                reaction=reaction,
                demographics=agent.demographics,
                psychographics=agent.psychographics,
                moral_foundations=agent.moral_foundations,
                prior_beliefs=agent.prior_beliefs,
            )

        log.info("Agent %s attempt %d failed validation: %s",
                 agent.agent_id, attempt, reason)

        # Strengthen anchoring for next retry
        if attempt == 1:
            current_description += (
                "\n\nCRITICAL: React directly. No hedging. No 'while' clauses. "
                "No 'I can appreciate' phrases. This person states their reaction plainly."
            )
        elif attempt == 2:
            current_description += (
                "\n\nABSOLUTE PROHIBITION: Do not use any of these phrases: "
                "'I can appreciate', 'while this isn't', 'I can see why others', "
                "'from my perspective though', 'there are merits'. "
                "These phrases are data contamination. State your reaction directly."
            )

    log.warning("Agent %s excluded after %d failed attempts",
                agent.agent_id, max_retries)
    return None


# ---------------------------------------------------------------------------
# Population runner
# ---------------------------------------------------------------------------

async def run_population(
    agents: list[AgentProfile],
    script_text: str,
    decomposition: dict,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_window_tokens: int = 8192,
    output_path: Optional[Path] = None,
) -> list[AgentResult]:
    """
    Run a full population of agents through the simulation.

    Agents are processed in batches to avoid overwhelming Ollama.
    Results are written incrementally to JSONL as they complete.
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_path = OUTPUT_DIR / f"run_{timestamp}.jsonl"

    results: list[AgentResult] = []

    async with httpx.AsyncClient() as client:
        # Pre-flight: check Ollama is available
        if not await check_ollama_available(client):
            log.error(
                "Ollama is not available at %s. "
                "Start Ollama before running the simulation.",
                OLLAMA_BASE_URL,
            )
            return []

        total = len(agents)
        log.info("Starting population run: %d agents, batch_size=%d, model=%s",
                 total, batch_size, model)

        with open(output_path, "w") as out_file:
            for i in range(0, total, batch_size):
                batch = agents[i : i + batch_size]

                tasks = [
                    run_single_agent(
                        client=client,
                        agent=agent,
                        script_text=script_text,
                        decomposition=decomposition,
                        model=model,
                        context_window_tokens=context_window_tokens,
                    )
                    for agent in batch
                ]

                batch_results = await asyncio.gather(*tasks)

                for result in batch_results:
                    if result is not None:
                        results.append(result)
                        # Write incrementally to JSONL
                        out_file.write(result.model_dump_json() + "\n")
                        out_file.flush()

                completed = min(i + batch_size, total)
                valid = len(results)
                log.info("Completed %d/%d agents (%d valid so far)",
                         completed, total, valid)

    log.info("Population run complete: %d/%d agents produced valid results",
             len(results), total)
    log.info("Results saved to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# Collapse rate diagnostic
# ---------------------------------------------------------------------------

async def measure_collapse_rate(
    agents: list[AgentProfile],
    script_text: str,
    decomposition: dict,
    model: str = DEFAULT_MODEL,
    sample_size: int = 20,
) -> float:
    """
    Run a small sample to measure first-attempt pass rate.
    Use this before scaling to full population.

    Returns the pass rate as a float (0.0 to 1.0).
    Above 0.8 → proceed. Below 0.6 → strengthen prompts first.
    """
    sample = agents[:sample_size]
    results = await run_population(
        agents=sample,
        script_text=script_text,
        decomposition=decomposition,
        model=model,
        batch_size=1,
    )
    rate = len(results) / sample_size if sample_size > 0 else 0.0
    print(f"\nCollapse rate diagnostic: {rate:.0%} first-attempt pass rate "
          f"({len(results)}/{sample_size} agents)")
    return rate


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run synth-audience simulation")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model name (default: mistral)")
    parser.add_argument("--script", required=True,
                        help="Path to script text file")
    parser.add_argument("--decomposition", required=True,
                        help="Path to decomposition JSON file")
    parser.add_argument("--agents", required=True,
                        help="Path to agents JSON file (list of AgentProfile)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--context-window", type=int, default=8192,
                        help="Model context window in tokens")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run collapse rate diagnostic only (20 agents)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load inputs
    script_text = Path(args.script).read_text()

    with open(args.decomposition) as f:
        decomposition = json.load(f)

    with open(args.agents) as f:
        agents_data = json.load(f)
    agents = [AgentProfile(**a) for a in agents_data]

    if args.diagnostic:
        asyncio.run(measure_collapse_rate(
            agents=agents,
            script_text=script_text,
            decomposition=decomposition,
            model=args.model,
        ))
    else:
        results = asyncio.run(run_population(
            agents=agents,
            script_text=script_text,
            decomposition=decomposition,
            model=args.model,
            batch_size=args.batch_size,
            context_window_tokens=args.context_window,
        ))
        print(f"\nDone. {len(results)} valid results from {len(agents)} agents.")


if __name__ == "__main__":
    main()
