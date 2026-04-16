"""
Pydantic models for agent profiles, demographic targets, and reaction schemas.

These are the data contracts that flow through the entire pipeline:
  DemographicTarget -> sampler -> AgentProfile -> prompts -> runner -> AgentReaction

The AgentReaction schema mirrors the JSON structure that the LLM must output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Demographic target — what the caller asks for when sampling agents
# ---------------------------------------------------------------------------

class DemographicTarget(BaseModel):
    age_min: int
    age_max: int
    gender: Optional[str] = None          # "male", "female"
    education: Optional[str] = None       # "no_college", "some_college", "college", "postgrad"
    geography: Optional[str] = None       # "urban", "suburban", "rural"
    region: Optional[str] = None          # "northeast", "south", "midwest", "west"
    income_bracket: Optional[str] = None
    political_lean: Optional[str] = None  # override ANES sampling if specified


# ---------------------------------------------------------------------------
# Content sensitivities — derived deterministically from profile traits
# ---------------------------------------------------------------------------

class ContentSensitivities(BaseModel):
    sex_work_depictions: str = "neutral"        # "high_salience" | "neutral" | "low_salience"
    class_dynamics: str = "neutral"
    female_agency: str = "neutral"
    pacing_tolerance: float = 0.5
    tonal_ambiguity_tolerance: float = 0.5


# ---------------------------------------------------------------------------
# Agent profile — the full synthetic respondent
# ---------------------------------------------------------------------------

class AgentProfile(BaseModel):
    agent_id: str
    demographics: dict                     # age, gender, education, geography, region, income
    psychographics: dict                   # Big Five: openness, conscientiousness, etc.
    moral_foundations: dict                 # care, fairness, loyalty, authority, purity, liberty
    media_diet: dict                       # primary_sources list, mainstream_trust float
    prior_beliefs: dict                    # political_lean float -1 to +1
    identity_salience: dict
    content_sensitivities: ContentSensitivities
    persona_anchors: str                   # 3-5 specific behavioral facts for prompt
    natural_language_description: str      # full rendered persona for LLM system prompt


# ---------------------------------------------------------------------------
# Reaction schema — what each agent must output as JSON
# ---------------------------------------------------------------------------

class ImmediateReaction(BaseModel):
    primary_emotion: str
    secondary_emotion: str
    intensity: int = Field(ge=1, le=10)
    gut_response: str


class OverallAssessment(BaseModel):
    valence: str                           # "positive" | "negative" | "mixed"
    confidence: int = Field(ge=1, le=10)
    would_watch_full_film: bool
    would_recommend_to_friend: bool
    expected_enjoyment_if_watched: int = Field(ge=1, le=10)


class ElementReaction(BaseModel):
    element_name: str
    reaction: str                          # "positive" | "negative" | "neutral"
    intensity: int = Field(ge=1, le=10)
    reason: str


class CharacterReaction(BaseModel):
    character: str
    reaction: str                          # "like" | "dislike" | "neutral" | "complex"
    reason: str


class FrictionPoint(BaseModel):
    what: str
    why: str


class AgentReaction(BaseModel):
    immediate_reaction: ImmediateReaction
    overall_assessment: OverallAssessment
    element_reactions: list[ElementReaction] = []
    character_reactions: list[CharacterReaction] = []
    friction_experienced: list[FrictionPoint] = []
    core_reason: str
    demographic_lens: str


# ---------------------------------------------------------------------------
# Full result record — agent metadata + reaction, saved to JSONL
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    agent_id: str
    reaction: AgentReaction
    demographics: dict
    psychographics: dict
    moral_foundations: dict
    prior_beliefs: dict
