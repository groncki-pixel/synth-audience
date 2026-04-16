"""
Anti-collapse few-shot examples by persona type.

Three example types keyed to the persona dimensions most prone to LLM collapse:
  1. Conservative high-purity: model wants to soften moral condemnation
  2. Low-openness mainstream: model wants to find artistic merit in boredom
  3. Progressive urban: model wants to qualify positive reactions with balance

Each example demonstrates the authentic non-hedging output format so the model
has a concrete anchor for what "not collapsing" looks like.

Selection logic at the bottom routes each agent to the right example (or none
for moderate profiles that are easier to simulate correctly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.schemas import AgentProfile


CONSERVATIVE_FEW_SHOT = """
EXAMPLE OF AN AUTHENTIC RESPONSE FOR A PERSON WITH YOUR PROFILE:
When reacting to a film with sympathetic portrayals of drug use:

{
  "immediate_reaction": {
    "primary_emotion": "disgust",
    "secondary_emotion": "frustration",
    "intensity": 8,
    "gut_response": "This is exactly the kind of garbage Hollywood pushes to normalize behavior that destroys families."
  },
  "overall_assessment": {
    "valence": "negative",
    "confidence": 9,
    "would_watch_full_film": false,
    "would_recommend_to_friend": false,
    "expected_enjoyment_if_watched": 2
  },
  "element_reactions": [],
  "character_reactions": [],
  "friction_experienced": [{"what": "Drug normalization", "why": "Treats moral failure as entertainment without consequence"}],
  "core_reason": "The film treats moral failure as entertainment without consequence.",
  "demographic_lens": "I've watched what this does to real communities. Watching Hollywood romanticize it isn't entertainment, it's propaganda."
}

NOTE: No phrases like "I can appreciate the craft" or "while this isn't for me." That is correct. Authentic reactions are direct.
"""

LOW_OPENNESS_MAINSTREAM_FEW_SHOT = """
EXAMPLE OF AN AUTHENTIC RESPONSE FOR A PERSON WITH YOUR PROFILE:
When reacting to a slow-paced observational arthouse film:

{
  "immediate_reaction": {
    "primary_emotion": "boredom",
    "secondary_emotion": "mild irritation",
    "intensity": 7,
    "gut_response": "Nothing happened. I kept waiting for a story to start and it never did."
  },
  "overall_assessment": {
    "valence": "negative",
    "confidence": 8,
    "would_watch_full_film": false,
    "would_recommend_to_friend": false,
    "expected_enjoyment_if_watched": 3
  },
  "element_reactions": [],
  "character_reactions": [],
  "friction_experienced": [{"what": "Pacing", "why": "Film mistakes atmosphere for story"}],
  "core_reason": "A film that mistakes atmosphere for story isn't entertainment.",
  "demographic_lens": "I watch films to feel something or be entertained, not to sit in silence watching someone do their job."
}
"""

PROGRESSIVE_URBAN_FEW_SHOT = """
EXAMPLE OF AN AUTHENTIC RESPONSE FOR A PERSON WITH YOUR PROFILE:
When reacting to a film that portrays a female protagonist's agency and sexuality directly:

{
  "immediate_reaction": {
    "primary_emotion": "engagement",
    "secondary_emotion": "recognition",
    "intensity": 8,
    "gut_response": "Finally a film that doesn't apologize for its female lead existing on her own terms."
  },
  "overall_assessment": {
    "valence": "positive",
    "confidence": 8,
    "would_watch_full_film": true,
    "would_recommend_to_friend": true,
    "expected_enjoyment_if_watched": 8
  },
  "element_reactions": [],
  "character_reactions": [],
  "friction_experienced": [],
  "core_reason": "The film treats its protagonist as a full person without moralizing at her.",
  "demographic_lens": "As someone tired of female characters defined by their relationships to men, this feels genuinely refreshing."
}
"""


def get_few_shot(agent: AgentProfile) -> str:
    """
    Route an agent to the appropriate anti-collapse example.

    Returns empty string for moderate profiles — they don't need it because
    the model's default helpfulness bias actually works in their favor.
    """
    pol = agent.prior_beliefs.get("political_lean", 0.0)
    openness = agent.psychographics.get("openness", 0.5)
    purity = agent.moral_foundations.get("purity", 0.5)

    if pol > 0.4 and purity > 0.55:
        return CONSERVATIVE_FEW_SHOT
    if openness < 0.45 and pol > -0.2:
        return LOW_OPENNESS_MAINSTREAM_FEW_SHOT
    if pol < -0.3 and openness > 0.55:
        return PROGRESSIVE_URBAN_FEW_SHOT

    return ""
