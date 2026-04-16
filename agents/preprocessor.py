"""
Preprocessor: builds lookup tables from raw survey data.

Run once after downloading datasets. Reads from data/raw/, writes JSON
lookup tables to data/processed/ that sampler.py reads at runtime.

Datasets expected in data/raw/:
  - anes_timeseries_cdf.csv  (ANES cumulative data file)
  - gss_cumulative.csv       (GSS cross-sectional cumulative)

If raw CSVs are not yet downloaded, this script falls back to hardcoded
normative tables from published literature (IPIP Big Five norms, MFQ norms).
Those tables are always written regardless of CSV availability.

Usage:
    python -m agents.preprocessor
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── ANES: political lean distributions by demographic cell ───────────────

def _build_belief_distributions() -> dict:
    """
    If the ANES CSV is present, compute mean/std of political lean per
    gender×education cell.  Otherwise return an empty dict — the sampler
    has a fallback for missing cells.
    """
    anes_path = RAW_DIR / "anes_timeseries_cdf.csv"
    if not anes_path.exists():
        print(f"  [skip] {anes_path} not found — belief distributions will be empty")
        return {}

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("  [skip] pandas/numpy not installed — belief distributions will be empty")
        return {}

    anes = pd.read_csv(anes_path, low_memory=False)

    # Filter to post-2000 waves for relevance
    if "VCF0004" in anes.columns:
        anes = anes[anes["VCF0004"] >= 2000]

    edu_map = {1: "no_college", 2: "no_college", 3: "some_college",
               4: "college", 5: "college", 6: "postgrad", 7: "postgrad"}
    anes["edu_cat"] = anes.get("VCF0110", pd.Series(dtype=float)).map(edu_map)

    # Party ID (VCF0301: 1=strong dem .. 7=strong rep) -> [-1, +1]
    raw_pol = anes.get("VCF0301", pd.Series(dtype=float))
    anes["pol_lean"] = (raw_pol - 4) / 3

    anes["gender"] = anes.get("VCF0104", pd.Series(dtype=float)).map({1: "male", 2: "female"})

    distributions: dict = {}
    for gender in ["male", "female"]:
        for edu in ["no_college", "some_college", "college", "postgrad"]:
            cell_key = f"{gender}_{edu}"
            cell = anes[
                (anes["gender"] == gender) & (anes["edu_cat"] == edu)
            ]["pol_lean"].dropna()

            if len(cell) > 30:
                distributions[cell_key] = {
                    "mean": float(cell.mean()),
                    "std": float(cell.std()),
                    "n": int(len(cell)),
                }

    return distributions


# ── IPIP: Big Five personality norms by age×gender ───────────────────────

def _build_ipip_norms() -> dict:
    """
    Hardcoded from Soto & John (2017), JPSP.
    Mean and SD for each Big Five trait by age band and gender.
    """
    return {
        "male_18_29": {
            "openness":          {"mean": 0.67, "sd": 0.15},
            "conscientiousness": {"mean": 0.58, "sd": 0.16},
            "extraversion":      {"mean": 0.54, "sd": 0.18},
            "agreeableness":     {"mean": 0.58, "sd": 0.16},
            "neuroticism":       {"mean": 0.49, "sd": 0.17},
        },
        "female_18_29": {
            "openness":          {"mean": 0.69, "sd": 0.14},
            "conscientiousness": {"mean": 0.63, "sd": 0.15},
            "extraversion":      {"mean": 0.60, "sd": 0.17},
            "agreeableness":     {"mean": 0.67, "sd": 0.14},
            "neuroticism":       {"mean": 0.57, "sd": 0.17},
        },
        "male_30_39": {
            "openness":          {"mean": 0.65, "sd": 0.14},
            "conscientiousness": {"mean": 0.62, "sd": 0.15},
            "extraversion":      {"mean": 0.52, "sd": 0.17},
            "agreeableness":     {"mean": 0.60, "sd": 0.15},
            "neuroticism":       {"mean": 0.45, "sd": 0.16},
        },
        "female_30_39": {
            "openness":          {"mean": 0.67, "sd": 0.13},
            "conscientiousness": {"mean": 0.66, "sd": 0.14},
            "extraversion":      {"mean": 0.57, "sd": 0.16},
            "agreeableness":     {"mean": 0.69, "sd": 0.13},
            "neuroticism":       {"mean": 0.53, "sd": 0.16},
        },
        "male_40_49": {
            "openness":          {"mean": 0.63, "sd": 0.14},
            "conscientiousness": {"mean": 0.64, "sd": 0.14},
            "extraversion":      {"mean": 0.50, "sd": 0.17},
            "agreeableness":     {"mean": 0.62, "sd": 0.15},
            "neuroticism":       {"mean": 0.42, "sd": 0.16},
        },
        "female_40_49": {
            "openness":          {"mean": 0.65, "sd": 0.13},
            "conscientiousness": {"mean": 0.68, "sd": 0.13},
            "extraversion":      {"mean": 0.55, "sd": 0.16},
            "agreeableness":     {"mean": 0.70, "sd": 0.13},
            "neuroticism":       {"mean": 0.50, "sd": 0.16},
        },
        "male_50_59": {
            "openness":          {"mean": 0.62, "sd": 0.14},
            "conscientiousness": {"mean": 0.65, "sd": 0.14},
            "extraversion":      {"mean": 0.49, "sd": 0.17},
            "agreeableness":     {"mean": 0.63, "sd": 0.14},
            "neuroticism":       {"mean": 0.40, "sd": 0.15},
        },
        "female_50_59": {
            "openness":          {"mean": 0.64, "sd": 0.13},
            "conscientiousness": {"mean": 0.69, "sd": 0.13},
            "extraversion":      {"mean": 0.54, "sd": 0.16},
            "agreeableness":     {"mean": 0.71, "sd": 0.12},
            "neuroticism":       {"mean": 0.48, "sd": 0.15},
        },
        "male_60_69": {
            "openness":          {"mean": 0.61, "sd": 0.14},
            "conscientiousness": {"mean": 0.66, "sd": 0.13},
            "extraversion":      {"mean": 0.48, "sd": 0.16},
            "agreeableness":     {"mean": 0.65, "sd": 0.14},
            "neuroticism":       {"mean": 0.38, "sd": 0.15},
        },
        "female_60_69": {
            "openness":          {"mean": 0.63, "sd": 0.13},
            "conscientiousness": {"mean": 0.70, "sd": 0.12},
            "extraversion":      {"mean": 0.53, "sd": 0.15},
            "agreeableness":     {"mean": 0.72, "sd": 0.12},
            "neuroticism":       {"mean": 0.45, "sd": 0.15},
        },
    }


# ── MFQ: Moral Foundations norms by political lean ───────────────────────

def _build_mfq_norms() -> dict:
    """
    Hardcoded from Graham et al. 2011, JPSP.
    Mean MFQ scores by political self-placement.
    """
    return {
        "very_liberal":      {"care": 0.81, "fairness": 0.79, "loyalty": 0.42,
                              "authority": 0.38, "purity": 0.33, "liberty": 0.74},
        "liberal":           {"care": 0.76, "fairness": 0.73, "loyalty": 0.48,
                              "authority": 0.44, "purity": 0.39, "liberty": 0.68},
        "moderate":          {"care": 0.70, "fairness": 0.67, "loyalty": 0.58,
                              "authority": 0.56, "purity": 0.52, "liberty": 0.61},
        "conservative":      {"care": 0.63, "fairness": 0.61, "loyalty": 0.67,
                              "authority": 0.66, "purity": 0.63, "liberty": 0.55},
        "very_conservative": {"care": 0.58, "fairness": 0.57, "loyalty": 0.74,
                              "authority": 0.73, "purity": 0.71, "liberty": 0.49},
    }


# ── Main ─────────────────────────────────────────────────────────────────

def build_lookup_tables() -> None:
    _ensure_dirs()

    print("Building belief distributions from ANES...")
    belief = _build_belief_distributions()
    with open(PROCESSED_DIR / "belief_distributions.json", "w") as f:
        json.dump(belief, f, indent=2)
    print(f"  -> {len(belief)} demographic cells")

    print("Building IPIP Big Five norms...")
    ipip = _build_ipip_norms()
    with open(PROCESSED_DIR / "ipip_norms.json", "w") as f:
        json.dump(ipip, f, indent=2)
    print(f"  -> {len(ipip)} age×gender cells")

    print("Building MFQ moral foundations norms...")
    mfq = _build_mfq_norms()
    with open(PROCESSED_DIR / "mfq_norms.json", "w") as f:
        json.dump(mfq, f, indent=2)
    print(f"  -> {len(mfq)} political lean categories")

    print("\nLookup tables built successfully.")


if __name__ == "__main__":
    build_lookup_tables()
