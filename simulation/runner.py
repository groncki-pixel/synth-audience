"""
Async batch agent runner with pluggable inference providers.

Orchestrates the full simulation: for each agent, builds prompts, calls the
configured provider (Ollama, Gemini, or Claude), parses JSON, validates
against collapse heuristics, retries on failure with strengthened anchoring.

Design patterns from OpenClaw reference:
  - pi-embedded-runner/run.ts: retry loop with failover classification,
    timeout handling, abort signals, and progressive retry escalation
  - compact.ts: batch execution with concurrent agent calls
  - context-window-guard.ts: pre-flight context size check before each call

Key constraints:
  - Provider abstraction allows local Ollama or hosted Gemini/Claude
  - Batch size defaults to 5 to avoid overwhelming local GPU
  - Every response goes through validator.py before acceptance
  - Failed validation triggers retry up to 3 times
  - Retry 2 strengthens persona anchoring in the system prompt
  - All outputs saved as JSONL to simulation/outputs/

Usage:
    python -m simulation.runner --provider ollama --model mistral \\
        --script content/scripts/anora_act1.txt \\
        --decomposition content/decompositions/anora_act1.json \\
        --agents agents/profiles/anora.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from agents.schemas import AgentProfile, AgentReaction, AgentResult
from simulation.context_guard import guard_context
from simulation.cost_tracker import CostTracker
from simulation.inference_config import AGENT_CONFIG, OLLAMA_BASE_URL
from simulation.prompts import (
    REACTION_PROMPT_TEMPLATE,
    build_system_prompt,
    build_user_prompt,
)
from simulation.providers import (
    ClaudeProvider,
    GeminiProvider,
    InferenceProvider,
    OllamaProvider,
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
# Ollama health check (kept for backward compat; only meaningful for Ollama)
# ---------------------------------------------------------------------------

async def check_ollama_available(provider: InferenceProvider) -> bool:
    """Verify Ollama is reachable. Returns True for non-Ollama providers."""
    if not isinstance(provider, OllamaProvider):
        return True
    return await provider.health_check()


# ---------------------------------------------------------------------------
# Backward-compatible thin wrapper around OllamaProvider
# ---------------------------------------------------------------------------

async def _call_ollama(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Backward-compatible wrapper used by external callers and older tests."""
    provider = OllamaProvider(model=model, client=client)
    return await provider.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=AGENT_CONFIG,
        timeout_seconds=timeout_seconds,
    )


def _extract_json(raw: str) -> Optional[dict]:
    """
    Extract JSON from raw model output. Handles cases where the model
    wraps JSON in markdown code fences or adds preamble text.
    """
    text = raw.strip()

    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    if brace_start == -1:
        return None

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
    agent: AgentProfile,
    script_text: str,
    decomposition: dict,
    provider: Optional[InferenceProvider] = None,
    context_window_tokens: int = 8192,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cost_tracker: Optional[CostTracker] = None,
) -> Optional[AgentResult]:
    """
    Run a single agent through the simulation with retry logic.

    Retry strategy (modeled on OpenClaw's run.ts retry loop):
      - Attempt 1: standard prompts
      - Attempt 2: add explicit anti-hedging reinforcement to persona
      - Attempt 3: further strengthen with direct prohibition list

    Returns AgentResult on success, None if all retries exhausted.
    """
    if provider is None:
        provider = OllamaProvider()

    decomposition_json = json.dumps(decomposition, indent=2)

    current_description = agent.natural_language_description

    for attempt in range(1, max_retries + 1):
        modified_agent = agent.model_copy()
        object.__setattr__(modified_agent, "natural_language_description", current_description)

        system_prompt = build_system_prompt(modified_agent)

        safe_script = guard_context(
            system_prompt=system_prompt,
            user_prompt_template=REACTION_PROMPT_TEMPLATE,
            script_text=script_text,
            decomposition_json=decomposition_json,
            context_window_tokens=context_window_tokens,
        )

        user_prompt = build_user_prompt(safe_script, decomposition_json)

        raw_response = await provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=AGENT_CONFIG,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
        if cost_tracker is not None and raw_response is not None:
            cost_tracker.record(
                provider=provider.name,
                model=provider.model,
                input_tokens=getattr(provider, "last_input_tokens", 0),
                output_tokens=getattr(provider, "last_output_tokens", 0),
            )
        if raw_response is None:
            log.warning("Agent %s attempt %d: no response from provider",
                        agent.agent_id, attempt)
            continue

        parsed = _extract_json(raw_response)
        if parsed is None:
            log.warning("Agent %s attempt %d: JSON parse failed",
                        agent.agent_id, attempt)
            continue

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
    provider: Optional[InferenceProvider] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_window_tokens: int = 8192,
    output_path: Optional[Path] = None,
    cost_path: Optional[Path] = None,
) -> list[AgentResult]:
    """
    Run a full population of agents through the simulation.

    Agents are processed in batches to avoid overwhelming the provider.
    Results are written incrementally to JSONL as they complete; a per-run
    cost summary is written alongside.
    """
    if provider is None:
        provider = OllamaProvider()

    timestamp = int(time.time())
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"run_{timestamp}.jsonl"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if cost_path is None:
        cost_path = output_path.parent / f"cost_{timestamp}.json"

    results: list[AgentResult] = []
    cost_tracker = CostTracker()

    try:
        if not await check_ollama_available(provider):
            log.error(
                "Ollama is not available at %s. "
                "Start Ollama before running the simulation.",
                OLLAMA_BASE_URL,
            )
            return []

        total = len(agents)
        log.info(
            "Starting population run: %d agents, batch_size=%d, provider=%s, model=%s",
            total, batch_size, provider.name, provider.model,
        )

        with open(output_path, "w") as out_file:
            for i in range(0, total, batch_size):
                batch = agents[i : i + batch_size]

                tasks = [
                    run_single_agent(
                        agent=agent,
                        script_text=script_text,
                        decomposition=decomposition,
                        provider=provider,
                        context_window_tokens=context_window_tokens,
                        cost_tracker=cost_tracker,
                    )
                    for agent in batch
                ]

                batch_results = await asyncio.gather(*tasks)

                for result in batch_results:
                    if result is not None:
                        results.append(result)
                        out_file.write(result.model_dump_json() + "\n")
                        out_file.flush()

                completed = min(i + batch_size, total)
                valid = len(results)
                log.info("Completed %d/%d agents (%d valid so far)",
                         completed, total, valid)
    finally:
        # Best-effort: close Ollama's owned client if we created it
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass

    cost_tracker.save(cost_path)
    log.info("Population run complete: %d/%d agents produced valid results",
             len(results), len(agents))
    log.info("Results saved to %s", output_path)
    log.info("Cost summary saved to %s", cost_path)

    return results


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def build_provider(name: str, model: Optional[str] = None) -> InferenceProvider:
    """Construct the InferenceProvider matching a CLI provider choice."""
    name = name.lower()
    if name == "ollama":
        return OllamaProvider(model=model or DEFAULT_MODEL)
    if name == "gemini":
        from simulation.providers import DEFAULT_GEMINI_MODEL
        return GeminiProvider(model=model or DEFAULT_GEMINI_MODEL)
    if name == "claude":
        from simulation.providers import DEFAULT_CLAUDE_MODEL
        return ClaudeProvider(model=model or DEFAULT_CLAUDE_MODEL)
    raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Collapse rate diagnostic
# ---------------------------------------------------------------------------

async def measure_collapse_rate(
    agents: list[AgentProfile],
    script_text: str,
    decomposition: dict,
    provider: Optional[InferenceProvider] = None,
    sample_size: int = 20,
) -> float:
    """Run a small sample to measure first-attempt pass rate."""
    sample = agents[:sample_size]
    results = await run_population(
        agents=sample,
        script_text=script_text,
        decomposition=decomposition,
        provider=provider,
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
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "gemini", "claude"],
                        help="Inference provider to use")
    parser.add_argument("--model", default=None,
                        help="Model name (defaults depend on provider)")
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

    script_text = Path(args.script).read_text()

    with open(args.decomposition) as f:
        decomposition = json.load(f)

    with open(args.agents) as f:
        agents_data = json.load(f)
    agents = [AgentProfile(**a) for a in agents_data]

    provider = build_provider(args.provider, args.model)

    if args.diagnostic:
        asyncio.run(measure_collapse_rate(
            agents=agents,
            script_text=script_text,
            decomposition=decomposition,
            provider=provider,
        ))
    else:
        results = asyncio.run(run_population(
            agents=agents,
            script_text=script_text,
            decomposition=decomposition,
            provider=provider,
            batch_size=args.batch_size,
            context_window_tokens=args.context_window,
        ))
        print(f"\nDone. {len(results)} valid results from {len(agents)} agents.")


if __name__ == "__main__":
    main()
