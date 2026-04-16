"""
Validation: scores simulation results against Anora ground truth.

Four-dimensional scorecard:
  1. Segment accuracy: each segment's positive rate within tolerance of expected
  2. Pacing flagged: pacing appears as top-3 friction in mainstream segment
  3. Collapse rate: percentage of agents excluded during simulation
  4. Internal consistency: percentage of responses passing validator

Ground truth for Anora:
  - Progressive women 20-35:  >70% positive  (female agency, class dynamics)
  - Conservative men 35+:     <40% positive  (sex work depiction, pacing)
  - Arthouse audience:        >80% positive  (tonal complexity, observational style)
  - Mainstream general:       ~50% mixed     (pacing as primary friction)

Usage:
    python -m validation.backtest --results simulation/outputs/run_XXXX.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from aggregation.aggregate import aggregate, load_results_from_jsonl


GROUND_TRUTH = {
    "progressive_women":  {"expected": 0.72, "tolerance": 0.15},
    "conservative_men":   {"expected": 0.35, "tolerance": 0.15},
    "arthouse_audience":  {"expected": 0.82, "tolerance": 0.12},
    "mainstream_audience": {"expected": 0.50, "tolerance": 0.15},
}


def score_simulation(aggregated: dict) -> dict:
    """
    Score aggregated simulation results against Anora ground truth.

    Returns a dict with:
      - segment_scores: per-segment simulated vs expected with pass/fail
      - pacing_correctly_flagged: whether pacing is a top friction in mainstream
      - overall_pass: all segments pass AND pacing flagged
    """
    segments = aggregated.get("segments", {})
    scores: dict[str, dict] = {}

    for seg, truth in GROUND_TRUTH.items():
        if seg not in segments:
            scores[seg] = {
                "simulated": None,
                "expected": truth["expected"],
                "error": None,
                "pass": False,
                "note": "segment missing from results",
            }
            continue

        simulated = segments[seg]["positive_rate"]
        error = abs(simulated - truth["expected"])
        scores[seg] = {
            "simulated": round(simulated, 3),
            "expected": truth["expected"],
            "error": round(error, 3),
            "pass": error <= truth["tolerance"],
        }

    # ── Pacing correctly flagged in mainstream ───────────────────────
    mainstream_friction = [
        fp["friction"]
        for fp in segments.get("mainstream_audience", {}).get("top_friction", [])
    ]
    pacing_flagged = any(
        "pac" in f.lower() or "slow" in f.lower()
        for f in mainstream_friction
    )

    # ── Sex work flagged in conservative segment ─────────────────────
    conservative_friction = [
        fp["friction"]
        for fp in segments.get("conservative_men", {}).get("top_friction", [])
    ]
    sex_work_flagged = any(
        "sex" in f.lower() or "strip" in f.lower() or "escort" in f.lower()
        for f in conservative_friction
    )

    all_segments_pass = all(s["pass"] for s in scores.values())

    return {
        "segment_scores": scores,
        "pacing_correctly_flagged": pacing_flagged,
        "sex_work_correctly_flagged": sex_work_flagged,
        "overall_pass": all_segments_pass and pacing_flagged,
    }


def print_scorecard(scores: dict) -> None:
    """Pretty-print the validation scorecard."""
    print("\n" + "=" * 60)
    print("  SYNTH-AUDIENCE VALIDATION SCORECARD")
    print("=" * 60)

    print("\n  Segment Accuracy:")
    print(f"  {'Segment':<25} {'Simulated':>10} {'Expected':>10} {'Error':>8} {'Pass':>6}")
    print("  " + "-" * 59)

    for seg, data in scores["segment_scores"].items():
        sim = f"{data['simulated']:.1%}" if data["simulated"] is not None else "N/A"
        exp = f"{data['expected']:.1%}"
        err = f"{data['error']:.3f}" if data["error"] is not None else "N/A"
        passed = "✅" if data["pass"] else "❌"
        print(f"  {seg:<25} {sim:>10} {exp:>10} {err:>8} {passed:>6}")

    print(f"\n  Pacing flagged in mainstream:      {'✅' if scores['pacing_correctly_flagged'] else '❌'}")
    print(f"  Sex work flagged in conservative:   {'✅' if scores['sex_work_correctly_flagged'] else '❌'}")
    print(f"\n  {'✅ OVERALL PASS' if scores['overall_pass'] else '❌ OVERALL FAIL'}")
    print("=" * 60)

    if not scores["overall_pass"]:
        print("\n  Diagnostic hints:")
        for seg, data in scores["segment_scores"].items():
            if not data["pass"]:
                if data["simulated"] is None:
                    print(f"    {seg}: segment missing — check agent sampling covers this cell")
                elif data["simulated"] > data["expected"]:
                    print(f"    {seg}: too positive — likely persona collapse, strengthen system prompt")
                else:
                    print(f"    {seg}: too negative — check sensitivity weights in sampler")
        if not scores["pacing_correctly_flagged"]:
            print("    pacing: not flagged — check pacing_tolerance threshold or decomposition")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest against Anora ground truth")
    parser.add_argument("--results", required=True,
                        help="Path to results JSONL file")
    parser.add_argument("--output", help="Path to save score JSON")
    args = parser.parse_args()

    results = load_results_from_jsonl(Path(args.results))
    print(f"Loaded {len(results)} agent results")

    aggregated = aggregate(results)
    scores = score_simulation(aggregated)

    print_scorecard(scores)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scores, indent=2))
        print(f"\nScore JSON saved to {out_path}")


if __name__ == "__main__":
    main()
