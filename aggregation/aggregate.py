"""
Aggregation: segment breakdowns, element-level attribution, and synthesis.

Takes raw AgentResult records from the simulation runner and produces:
  1. Overall population statistics
  2. Per-segment breakdowns (progressive women, conservative men, arthouse, mainstream)
  3. Element-level attribution (which script elements drive positive/negative reactions)
  4. Top friction and resonance points per segment

The segment definitions are configurable but ship with Anora-specific defaults
for the proof-of-concept validation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from agents.schemas import AgentResult


# ---------------------------------------------------------------------------
# Segment definitions — callables that filter agent results
# ---------------------------------------------------------------------------

ANORA_SEGMENTS: dict[str, Callable[[AgentResult], bool]] = {
    "progressive_women": lambda r: (
        r.prior_beliefs.get("political_lean", 0) < -0.3
        and r.demographics.get("gender") == "female"
        and r.demographics.get("age", 99) < 40
    ),
    "conservative_men": lambda r: (
        r.prior_beliefs.get("political_lean", 0) > 0.3
        and r.demographics.get("gender") == "male"
        and r.demographics.get("age", 0) > 35
    ),
    "arthouse_audience": lambda r: (
        r.psychographics.get("openness", 0) > 0.65
    ),
    "mainstream_audience": lambda r: (
        r.psychographics.get("openness", 0.5) < 0.5
        and r.prior_beliefs.get("political_lean", 0) > -0.2
    ),
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    results: list[AgentResult],
    segments: Optional[dict[str, Callable[[AgentResult], bool]]] = None,
) -> dict:
    """
    Aggregate simulation results into overall + per-segment breakdowns.

    Args:
        results: List of valid AgentResult records.
        segments: Segment filter functions. Defaults to ANORA_SEGMENTS.

    Returns:
        Dict with 'overall', 'segments', and 'element_attribution' keys.
    """
    if segments is None:
        segments = ANORA_SEGMENTS

    if not results:
        return {"overall": {}, "segments": {}, "element_attribution": {}}

    # ── Overall stats ────────────────────────────────────────────────
    valences = [r.reaction.overall_assessment.valence for r in results]
    enjoyment = [r.reaction.overall_assessment.expected_enjoyment_if_watched for r in results]

    overall = {
        "n": len(results),
        "positive_rate": valences.count("positive") / len(valences),
        "negative_rate": valences.count("negative") / len(valences),
        "mixed_rate": valences.count("mixed") / len(valences),
        "mean_enjoyment": float(np.mean(enjoyment)),
        "watch_rate": float(np.mean([
            r.reaction.overall_assessment.would_watch_full_film for r in results
        ])),
    }

    # ── Per-segment breakdowns ───────────────────────────────────────
    segment_results: dict[str, dict] = {}

    for seg_name, filter_fn in segments.items():
        seg = [r for r in results if filter_fn(r)]
        if not seg:
            continue

        seg_valences = [r.reaction.overall_assessment.valence for r in seg]
        seg_enjoyment = [r.reaction.overall_assessment.expected_enjoyment_if_watched for r in seg]

        # Top friction points
        friction_counter: Counter = Counter()
        friction_reasons: dict[str, list[str]] = defaultdict(list)
        for r in seg:
            for fp in r.reaction.friction_experienced:
                friction_counter[fp.what] += 1
                friction_reasons[fp.what].append(fp.why)

        top_friction = [
            {
                "friction": item,
                "frequency": count / len(seg),
                "sample_reasons": friction_reasons[item][:3],
            }
            for item, count in friction_counter.most_common(3)
        ]

        segment_results[seg_name] = {
            "n": len(seg),
            "positive_rate": seg_valences.count("positive") / len(seg_valences),
            "negative_rate": seg_valences.count("negative") / len(seg_valences),
            "mixed_rate": seg_valences.count("mixed") / len(seg_valences),
            "mean_enjoyment": float(np.mean(seg_enjoyment)),
            "watch_rate": float(np.mean([
                r.reaction.overall_assessment.would_watch_full_film for r in seg
            ])),
            "top_friction": top_friction,
            "sample_core_reasons": [r.reaction.core_reason for r in seg[:5]],
        }

    # ── Element-level attribution ────────────────────────────────────
    element_scores: dict[str, dict] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "reasons": []}
    )

    for r in results:
        for er in r.reaction.element_reactions:
            el = er.element_name
            element_scores[el][er.reaction] = element_scores[el].get(er.reaction, 0) + 1
            element_scores[el]["total"] += 1
            if er.reason:
                element_scores[el]["reasons"].append(er.reason)

    # Compute net scores
    for el in element_scores:
        t = element_scores[el]["total"]
        if t > 0:
            element_scores[el]["net_score"] = (
                (element_scores[el]["positive"] - element_scores[el]["negative"]) / t
            )

    return {
        "overall": overall,
        "segments": segment_results,
        "element_attribution": dict(element_scores),
    }


# ---------------------------------------------------------------------------
# Load results from JSONL
# ---------------------------------------------------------------------------

def load_results_from_jsonl(path: Path) -> list[AgentResult]:
    """Load AgentResult records from a JSONL file."""
    results: list[AgentResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            results.append(AgentResult(**data))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate simulation results")
    parser.add_argument("--results", required=True,
                        help="Path to results JSONL file")
    parser.add_argument("--output", help="Path to save aggregation JSON")
    args = parser.parse_args()

    results = load_results_from_jsonl(Path(args.results))
    aggregated = aggregate(results)

    output = json.dumps(aggregated, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Aggregation saved to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
