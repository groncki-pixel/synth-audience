"""
One-command end-to-end pipeline.

Runs: sample agents -> decompose script -> simulate -> aggregate -> validate.

Examples:
    python scripts/pipeline.py --script content/scripts/sample.txt --dry-run
    python scripts/pipeline.py --film anora --script content/scripts/anora.txt \\
        --provider gemini --n-agents 100

The default demographic spread is hardcoded for now (mix of genders,
educations, and age ranges); making this configurable is tracked for the
next session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow `python scripts/pipeline.py` from any cwd without requiring PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.sampler import sample_agents
from agents.schemas import AgentProfile, DemographicTarget
from aggregation.aggregate import aggregate
from simulation.runner import build_provider, run_population

log = logging.getLogger("scripts.pipeline")


# A reasonable default mix to spread across genders, education, ages, and
# political leans. Each entry's count is a rough share of the total agent
# pool; the sum need not equal n_agents — we scale proportionally.
DEFAULT_DEMOGRAPHIC_MIX = [
    {"weight": 1, "gender": "female", "education": "college",
     "age_min": 22, "age_max": 34, "geography": "urban", "region": "northeast"},
    {"weight": 1, "gender": "female", "education": "some_college",
     "age_min": 35, "age_max": 54, "geography": "suburban", "region": "midwest"},
    {"weight": 1, "gender": "female", "education": "no_college",
     "age_min": 30, "age_max": 60, "geography": "rural", "region": "south"},
    {"weight": 1, "gender": "female", "education": "postgrad",
     "age_min": 28, "age_max": 50, "geography": "urban", "region": "west"},
    {"weight": 1, "gender": "male", "education": "college",
     "age_min": 22, "age_max": 34, "geography": "urban", "region": "northeast"},
    {"weight": 1, "gender": "male", "education": "some_college",
     "age_min": 35, "age_max": 54, "geography": "suburban", "region": "midwest"},
    {"weight": 1, "gender": "male", "education": "no_college",
     "age_min": 30, "age_max": 60, "geography": "rural", "region": "south"},
    {"weight": 1, "gender": "male", "education": "postgrad",
     "age_min": 28, "age_max": 50, "geography": "urban", "region": "west"},
]


PROVIDER_DEFAULT_MODEL = {
    "ollama": "mistral",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-haiku-4-5",
}


def _allocate(n_total: int, weights: list[int]) -> list[int]:
    """Spread n_total across len(weights) buckets proportional to weights."""
    total_w = sum(weights)
    counts = [int(n_total * w / total_w) for w in weights]
    # Distribute the remainder to the first buckets so the sum matches n_total.
    deficit = n_total - sum(counts)
    for i in range(deficit):
        counts[i % len(counts)] += 1
    return counts


def sample_default_population(n_agents: int) -> list[AgentProfile]:
    counts = _allocate(n_agents, [b["weight"] for b in DEFAULT_DEMOGRAPHIC_MIX])
    population: list[AgentProfile] = []
    for bucket, n in zip(DEFAULT_DEMOGRAPHIC_MIX, counts):
        if n == 0:
            continue
        target = DemographicTarget(
            age_min=bucket["age_min"],
            age_max=bucket["age_max"],
            gender=bucket["gender"],
            education=bucket["education"],
            geography=bucket["geography"],
            region=bucket["region"],
            income_bracket="middle",
        )
        population.extend(sample_agents(target, n_agents=n))
    return population


def _save_agents(agents: list[AgentProfile], path: Path) -> None:
    path.write_text(json.dumps([a.model_dump() for a in agents], indent=2))


def _print_summary(
    *,
    film: str,
    n_agents: int,
    n_valid: int,
    aggregation: dict,
    cost_path: Optional[Path],
    dry_run: bool = False,
) -> None:
    print()
    print("=" * 60)
    print(f"Pipeline summary: {film}")
    print("=" * 60)
    print(f"Agents sampled : {n_agents}")
    if dry_run:
        print("Mode           : dry-run (no LLM calls)")
        print("=" * 60)
        return
    print(f"Valid results  : {n_valid}")
    if n_agents:
        collapse_rate = 1.0 - n_valid / n_agents
        print(f"Collapse rate  : {collapse_rate:.1%}")

    if cost_path and cost_path.exists():
        try:
            cost = json.loads(cost_path.read_text())
            print(f"Total cost USD : ${cost.get('total_cost_usd', 0.0):.4f}")
            print(f"Tokens in/out  : "
                  f"{cost.get('total_input_tokens', 0):,} / "
                  f"{cost.get('total_output_tokens', 0):,}")
        except (OSError, json.JSONDecodeError):
            pass

    overall = aggregation.get("overall", {}) if aggregation else {}
    if overall:
        print()
        print("Overall valence:")
        print(f"  positive   : {overall.get('positive_rate', 0):.1%}")
        print(f"  negative   : {overall.get('negative_rate', 0):.1%}")
        print(f"  mixed      : {overall.get('mixed_rate', 0):.1%}")
        print(f"  watch rate : {overall.get('watch_rate', 0):.1%}")

    segments = aggregation.get("segments", {}) if aggregation else {}
    if segments:
        print()
        print("Top segments:")
        for name, data in segments.items():
            print(f"  {name:24s} n={data['n']:3d}  "
                  f"pos={data['positive_rate']:.0%}  "
                  f"watch={data['watch_rate']:.0%}")

        print()
        print("Top friction points:")
        for name, data in segments.items():
            top = data.get("top_friction", [])
            if not top:
                continue
            top_friction = top[0]
            print(f"  {name:24s} -> {top_friction['friction']} "
                  f"({top_friction['frequency']:.0%})")
    print("=" * 60)


async def _run_async(args: argparse.Namespace) -> int:
    script_path = Path(args.script)
    script_text = script_path.read_text()
    film = args.film or script_path.stem

    timestamp = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("outputs") / f"{film}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pipeline output directory: %s", output_dir)

    log.info("Sampling %d agents...", args.n_agents)
    agents = sample_default_population(args.n_agents)
    agents_path = output_dir / "agents.json"
    _save_agents(agents, agents_path)
    log.info("Saved %d agents to %s", len(agents), agents_path)

    decomposition_path = output_dir / "decomposition.json"
    decomposition: dict = {}
    if args.skip_decomposition or args.dry_run:
        if args.dry_run:
            log.info("[dry-run] skipping decomposition")
        else:
            log.info("Skipping decomposition (--skip-decomposition)")
        decomposition = {"_note": "decomposition skipped"}
        decomposition_path.write_text(json.dumps(decomposition, indent=2))
    else:
        from content.decomposer import decompose_content
        decomposition = await decompose_content(
            script_text=script_text,
            title=film,
            model=args.model or PROVIDER_DEFAULT_MODEL[args.provider],
            save_dir=output_dir,
        )

    if args.dry_run:
        log.info("[dry-run] skipping simulation, aggregation, validation")
        _print_summary(
            film=film,
            n_agents=len(agents),
            n_valid=0,
            aggregation={},
            cost_path=None,
            dry_run=True,
        )
        return 0

    provider = build_provider(args.provider, args.model)

    results_path = output_dir / "results.jsonl"
    cost_path = output_dir / f"cost_{timestamp}.json"

    log.info("Running simulation with provider=%s model=%s",
             provider.name, provider.model)
    results = await run_population(
        agents=agents,
        script_text=script_text,
        decomposition=decomposition,
        provider=provider,
        output_path=results_path,
        cost_path=cost_path,
    )

    aggregation = aggregate(results)
    aggregation_path = output_dir / "aggregation.json"
    aggregation_path.write_text(json.dumps(aggregation, indent=2, default=str))
    log.info("Aggregation saved to %s", aggregation_path)

    _print_summary(
        film=film,
        n_agents=len(agents),
        n_valid=len(results),
        aggregation=aggregation,
        cost_path=cost_path,
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One-command pipeline runner")
    parser.add_argument("--film", help="Film title used for naming outputs")
    parser.add_argument("--script", required=True, help="Path to script text")
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument(
        "--provider", default="gemini", choices=["ollama", "gemini", "claude"]
    )
    parser.add_argument("--model", default=None,
                        help="Model name (defaults depend on provider)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/{film}_{timestamp}/)")
    parser.add_argument("--skip-decomposition", action="store_true",
                        help="Skip the decomposition step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sample + decompose only; no LLM agent calls")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
