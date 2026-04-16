"""
Agent sampler: generates AgentProfile instances from survey-derived distributions.

Takes a DemographicTarget and produces N agents whose personality traits,
moral foundations, media diets, and content sensitivities are sampled from
real population data (lookup tables built by preprocessor.py).

Each agent gets:
  - Big Five personality from IPIP norms for their age×gender cell
  - Political lean from ANES distributions for their gender×education cell
  - Moral foundations from MFQ norms keyed by political lean
  - Content sensitivities derived deterministically from the above
  - Persona anchors: 3-5 hyper-specific behavioral facts
  - Natural language description: full rendered persona for LLM system prompt
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import truncnorm

from agents.schemas import AgentProfile, ContentSensitivities, DemographicTarget

PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Load lookup tables (built by preprocessor.py)
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict:
    path = PROCESSED_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


BELIEF_DISTRIBUTIONS: dict = _load_json("belief_distributions.json")
IPIP_NORMS: dict = _load_json("ipip_norms.json")
MFQ_NORMS: dict = _load_json("mfq_norms.json")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_truncated_normal(mean: float, sd: float,
                              low: float = 0.0, high: float = 1.0) -> float:
    if sd <= 0:
        return float(np.clip(mean, low, high))
    a, b = (low - mean) / sd, (high - mean) / sd
    return float(truncnorm.rvs(a, b, loc=mean, scale=sd))


def _resolve_ipip_key(gender: Optional[str], age: int) -> str:
    """Map age to the decade-band key used in IPIP norms."""
    g = gender or "male"
    decade_start = (age // 10) * 10
    decade_end = min(decade_start + 9, 99)
    key = f"{g}_{decade_start}_{decade_end}"
    if key in IPIP_NORMS:
        return key
    # Fallback: closest available
    return f"{g}_30_39"


def _resolve_mfq_key(political_lean: float) -> str:
    if political_lean < -0.6:
        return "very_liberal"
    if political_lean < -0.2:
        return "liberal"
    if political_lean < 0.2:
        return "moderate"
    if political_lean < 0.6:
        return "conservative"
    return "very_conservative"


# ---------------------------------------------------------------------------
# Media diet (simplified — will be enriched with Pew data later)
# ---------------------------------------------------------------------------

def _sample_media_diet(target: DemographicTarget,
                        political_lean: float,
                        noise: float) -> dict:
    if political_lean < -0.3:
        sources = ["NPR", "NYT", "social media", "streaming"]
        trust = _sample_truncated_normal(0.62, 0.15 * (1 + noise))
    elif political_lean > 0.3:
        sources = ["Fox News", "talk radio", "Facebook", "local news"]
        trust = _sample_truncated_normal(0.35, 0.15 * (1 + noise))
    else:
        sources = ["local news", "social media", "streaming"]
        trust = _sample_truncated_normal(0.48, 0.15 * (1 + noise))

    return {
        "primary_sources": sources,
        "mainstream_trust": float(trust),
    }


# ---------------------------------------------------------------------------
# Content sensitivities — the bridge between profile and content reaction
# ---------------------------------------------------------------------------

def _derive_content_sensitivities(big5: dict, mfq: dict,
                                    political_lean: float) -> ContentSensitivities:
    def classify(score: float, low: float = 0.35, high: float = 0.65) -> str:
        if score > high:
            return "high_salience"
        if score < low:
            return "low_salience"
        return "neutral"

    purity_score = (mfq["purity"] * 0.5 + mfq["authority"] * 0.3 +
                    max(political_lean, 0) * 0.2)
    class_score = (mfq["care"] * 0.4 + mfq["fairness"] * 0.4 +
                   max(-political_lean, 0) * 0.2)
    agency_score = big5["openness"] * 0.4 + mfq["care"] * 0.3 + mfq["fairness"] * 0.3
    pacing_tol = big5["openness"] * 0.6 + big5["conscientiousness"] * 0.2
    tonal_tol = (big5["openness"] * 0.5 +
                 (1 - big5["conscientiousness"]) * 0.3 +
                 big5["neuroticism"] * 0.2)

    return ContentSensitivities(
        sex_work_depictions=classify(purity_score),
        class_dynamics=classify(class_score),
        female_agency=classify(agency_score),
        pacing_tolerance=float(pacing_tol),
        tonal_ambiguity_tolerance=float(tonal_tol),
    )


# ---------------------------------------------------------------------------
# Identity salience
# ---------------------------------------------------------------------------

def _build_identity_salience(target: DemographicTarget,
                              big5: dict,
                              political_lean: float) -> dict:
    salience: dict = {}
    if target.gender == "female" and big5["openness"] > 0.55:
        salience["gender_identity"] = "high"
    if abs(political_lean) > 0.5:
        salience["political_identity"] = "high"
    if target.geography == "rural":
        salience["place_identity"] = "moderate"
    return salience


# ---------------------------------------------------------------------------
# Persona anchors — specific behavioral facts that fight abstraction
# ---------------------------------------------------------------------------

def _generate_persona_anchors(big5: dict, mfq: dict, political_lean: float,
                                sensitivities: ContentSensitivities,
                                media_diet: dict) -> str:
    anchors: list[str] = []

    # Film taste
    if big5["openness"] < 0.35:
        anchors.append(
            "This person's favorite films are mainstream blockbusters and genre entertainment. "
            "They last stopped watching a film because 'nothing was happening.'"
        )
    elif big5["openness"] > 0.70:
        anchors.append(
            "This person's favorite films include slow, character-driven European cinema. "
            "They find most Hollywood films intellectually insulting."
        )

    # Sex work
    if mfq["purity"] > 0.65 and political_lean > 0.3:
        anchors.append(
            "This person finds graphic depictions of sex work in mainstream entertainment "
            "gratuitous and morally wrong. They would turn this off if watching at home."
        )
    elif mfq["purity"] < 0.35 and political_lean < -0.3:
        anchors.append(
            "This person has no moral discomfort with sex work depictions "
            "and would find moralizing about it patronizing."
        )

    # Pacing
    if sensitivities.pacing_tolerance < 0.35:
        anchors.append(
            "This person checks their phone within 10 minutes if a film hasn't hooked them. "
            "They described a recent slow film as 'an endurance test, not entertainment.'"
        )

    # Media trust
    if media_diet.get("mainstream_trust", 0.5) < 0.3:
        anchors.append(
            "This person deeply distrusts critical consensus. "
            "If critics universally love something, they assume it has been approved "
            "by people with an agenda."
        )

    # Class lens
    if mfq["fairness"] > 0.65 and political_lean < -0.2:
        anchors.append(
            "This person is acutely sensitive to class dynamics in storytelling. "
            "Stories about wealth disparity feel personally relevant to them."
        )

    return "\n".join(f"- {a}" for a in anchors)


# ---------------------------------------------------------------------------
# Natural language persona description — rendered into system prompt
# ---------------------------------------------------------------------------

def _render_natural_language(target: DemographicTarget, age: int,
                              big5: dict, mfq: dict, political_lean: float,
                              media_diet: dict, identity_salience: dict,
                              sensitivities: ContentSensitivities) -> str:
    # Political lean text
    if political_lean < -0.6:
        pol_text = "strongly progressive"
    elif political_lean < -0.3:
        pol_text = "moderately progressive"
    elif political_lean < 0.3:
        pol_text = "politically moderate"
    elif political_lean < 0.6:
        pol_text = "moderately conservative"
    else:
        pol_text = "strongly conservative"

    # Personality traits
    traits: list[str] = []
    if big5["openness"] > 0.65:
        traits.append("intellectually curious and open to new experiences")
    if big5["openness"] < 0.35:
        traits.append("practical and conventional in your tastes")
    if big5["conscientiousness"] > 0.65:
        traits.append("organized and detail-oriented")
    if big5["agreeableness"] > 0.65:
        traits.append("empathetic and cooperative")
    if big5["neuroticism"] > 0.65:
        traits.append("emotionally reactive and sensitive")
    trait_str = ", ".join(traits) if traits else "fairly balanced across personality dimensions"

    # Sensitivity text
    sex_work_map = {
        "high_salience": "Sex work is something you view with moral concern and discomfort.",
        "neutral": "Sex work is something you view pragmatically, without strong moral judgment.",
        "low_salience": "Sex work is something you view as legitimate labor deserving of respect.",
    }
    sex_work_text = sex_work_map[sensitivities.sex_work_depictions]

    if sensitivities.pacing_tolerance < 0.35:
        pacing_text = "Slow, observational storytelling frustrates you — you want narrative momentum."
    elif sensitivities.pacing_tolerance > 0.65:
        pacing_text = "You have genuine patience for slow, character-driven storytelling."
    else:
        pacing_text = "You prefer a moderate pace — you'll tolerate slow stretches if the payoff is there."

    sources = ", ".join(media_diet.get("primary_sources", ["mainstream TV", "social media"]))
    trust_text = ("trust mainstream media sources"
                  if media_diet.get("mainstream_trust", 0.5) > 0.5
                  else "are skeptical of mainstream media")

    fairness_line = (" You care deeply about fairness and class inequality."
                     if mfq["fairness"] > 0.65 else "")
    purity_line = (" You have strong views about moral tradition and purity."
                   if mfq["purity"] > 0.65 else "")
    liberty_line = (" You value personal liberty and distrust institutional authority."
                    if mfq["liberty"] > 0.65 else "")

    class_line = ("Class dynamics and wealth disparity in storytelling "
                  "resonate strongly with you."
                  if sensitivities.class_dynamics == "high_salience" else "")

    geography = target.geography or "suburban"
    region = target.region or "United States"
    education = target.education or "some_college"
    income = target.income_bracket or "middle"

    return (
        f"You are a {age}-year-old {target.gender} living in a {geography} area "
        f"of the {region}. You have a {education} level of education and earn "
        f"in the {income} income range.\n\n"
        f"Politically you are {pol_text}.{fairness_line}{purity_line}{liberty_line}\n\n"
        f"Your personality: you are {trait_str}.\n\n"
        f"Your media diet: you primarily consume {sources}. "
        f"You {trust_text}.\n\n"
        f"{sex_work_text}"
        f"{' ' + class_line if class_line else ''}\n\n"
        f"{pacing_text}"
    )


# ---------------------------------------------------------------------------
# Public API: sample N agents for a demographic target
# ---------------------------------------------------------------------------

def sample_agents(target: DemographicTarget,
                  n_agents: int,
                  noise_level: float = 0.12) -> list[AgentProfile]:
    agents: list[AgentProfile] = []

    for _ in range(n_agents):
        age = int(np.random.randint(target.age_min, target.age_max + 1))

        # Big Five from IPIP norms
        ipip_key = _resolve_ipip_key(target.gender, age)
        ipip = IPIP_NORMS.get(ipip_key, IPIP_NORMS.get("male_30_39", {}))

        big5 = {}
        for trait in ["openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"]:
            norms = ipip.get(trait, {"mean": 0.5, "sd": 0.15})
            big5[trait] = _sample_truncated_normal(
                norms["mean"],
                norms["sd"] * (1 + noise_level),
            )

        # Political lean from ANES
        belief_key = f"{target.gender}_{target.education}"
        if belief_key in BELIEF_DISTRIBUTIONS:
            dist = BELIEF_DISTRIBUTIONS[belief_key]
            political_lean = float(np.clip(
                np.random.normal(dist["mean"], dist["std"] * (1 + noise_level)),
                -1, 1,
            ))
        else:
            political_lean = float(np.random.normal(0, 0.4))
            political_lean = float(np.clip(political_lean, -1, 1))

        # MFQ from norms
        mfq_key = _resolve_mfq_key(political_lean)
        mfq_base = MFQ_NORMS.get(mfq_key, MFQ_NORMS.get("moderate", {}))

        mfq = {}
        for trait in ["care", "fairness", "loyalty", "authority", "purity", "liberty"]:
            base_val = mfq_base.get(trait, 0.5)
            mfq[trait] = _sample_truncated_normal(
                base_val,
                0.12 * (1 + noise_level),
            )

        # Derived layers
        media_diet = _sample_media_diet(target, political_lean, noise_level)
        sensitivities = _derive_content_sensitivities(big5, mfq, political_lean)
        identity_salience = _build_identity_salience(target, big5, political_lean)
        anchors = _generate_persona_anchors(big5, mfq, political_lean,
                                             sensitivities, media_diet)
        nl_desc = _render_natural_language(target, age, big5, mfq, political_lean,
                                            media_diet, identity_salience, sensitivities)

        agent = AgentProfile(
            agent_id=str(uuid.uuid4())[:8],
            demographics={
                "age": age,
                "gender": target.gender,
                "education": target.education,
                "geography": target.geography,
                "region": target.region,
                "income_bracket": target.income_bracket,
            },
            psychographics=big5,
            moral_foundations=mfq,
            media_diet=media_diet,
            prior_beliefs={"political_lean": political_lean},
            identity_salience=identity_salience,
            content_sensitivities=sensitivities,
            persona_anchors=anchors,
            natural_language_description=nl_desc,
        )
        agents.append(agent)

    return agents
