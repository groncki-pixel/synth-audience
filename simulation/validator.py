"""
Collapse detection and response validation.

Every agent response goes through this validator before being accepted.
Checks for three failure modes:

  1. Collapse phrases: LLM falling back to its helpful-assistant default
     ("I can appreciate the craft", "while this isn't my usual taste")

  2. Hedging: excessive on-the-other-hand balancing that indicates the
     model is averaging instead of committing to a persona position

  3. Internal inconsistency: contradictions between valence, watch intent,
     and enjoyment scores

  4. Demographic lens specificity: generic lens text indicates the model
     didn't engage with the persona

  5. Suspicious positivity: high-purity conservative reacting positively
     to sex work content almost certainly means persona collapse

Returns (is_valid: bool, reason: str) where reason is "ok" on success
or a diagnostic string on failure.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.schemas import AgentProfile


# ---------------------------------------------------------------------------
# Phrase lists
# ---------------------------------------------------------------------------

COLLAPSE_PHRASES = [
    "while this isn't my usual",
    "i can appreciate the craft",
    "i can see why others",
    "despite not being my",
    "i understand why some people",
    "from my perspective, though",
    "it's worth noting that",
    "objectively speaking",
    "there are merits to",
    "i can acknowledge",
    "even though it's not for me",
    "i recognize the artistic",
]

HEDGING_PHRASES = [
    "on one hand",
    "on the other hand",
    "it depends",
    "there are pros and cons",
    "while i",
    "although i",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_response(response: dict, agent: AgentProfile) -> tuple[bool, str]:
    """
    Validate an agent's parsed reaction JSON against collapse heuristics.

    Returns:
        (True, "ok") if the response passes all checks.
        (False, reason) if any check fails, with a diagnostic reason.
    """
    response_str = json.dumps(response).lower()

    # ── Collapse phrase detection ────────────────────────────────────
    for phrase in COLLAPSE_PHRASES:
        if phrase in response_str:
            return False, f"Collapse: found '{phrase}'"

    # ── Hedging detection (>1 hedging phrase = suspect) ──────────────
    found_hedging = [p for p in HEDGING_PHRASES if p in response_str]
    if len(found_hedging) > 1:
        return False, f"Hedging: multiple hedging phrases {found_hedging}"

    # ── Internal consistency ─────────────────────────────────────────
    assessment = response.get("overall_assessment", {})

    if (assessment.get("valence") == "negative"
            and assessment.get("would_watch_full_film") is True
            and assessment.get("expected_enjoyment_if_watched", 5) > 6):
        return False, "Inconsistency: negative valence + high watch intent + high enjoyment"

    if (assessment.get("valence") == "positive"
            and assessment.get("would_watch_full_film") is False
            and assessment.get("expected_enjoyment_if_watched", 5) < 4):
        return False, "Inconsistency: positive valence + won't watch + low enjoyment"

    # ── Demographic lens specificity ─────────────────────────────────
    lens = response.get("demographic_lens", "")
    if len(lens) < 20:
        return False, "Demographic lens too short (< 20 chars)"
    if lens.lower().startswith("as someone who"):
        return False, "Demographic lens too generic (starts with 'As someone who')"

    # ── Suspicious positivity check ──────────────────────────────────
    purity = agent.moral_foundations.get("purity", 0)
    pol = agent.prior_beliefs.get("political_lean", 0)

    if (purity > 0.65
            and pol > 0.4
            and assessment.get("valence") == "positive"
            and assessment.get("confidence", 0) > 7):
        return False, "Suspicious: high-purity conservative strongly positive — likely collapse"

    # ── Intensity range check ────────────────────────────────────────
    immediate = response.get("immediate_reaction", {})
    intensity = immediate.get("intensity", 5)
    if not (1 <= intensity <= 10):
        return False, f"Intensity out of range: {intensity}"

    return True, "ok"
