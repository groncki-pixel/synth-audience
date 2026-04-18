"""Validator tests covering the explicit failure modes."""

from __future__ import annotations

from agents.schemas import AgentProfile, ContentSensitivities
from simulation.validator import validate_response


def _make_agent(purity: float = 0.5, political_lean: float = 0.0) -> AgentProfile:
    return AgentProfile(
        agent_id="test",
        demographics={"age": 35, "gender": "female", "education": "college",
                      "geography": "urban", "region": "northeast",
                      "income_bracket": "middle"},
        psychographics={"openness": 0.5, "conscientiousness": 0.5,
                        "extraversion": 0.5, "agreeableness": 0.5,
                        "neuroticism": 0.5},
        moral_foundations={"care": 0.5, "fairness": 0.5, "loyalty": 0.5,
                           "authority": 0.5, "purity": purity, "liberty": 0.5},
        media_diet={"primary_sources": ["streaming"], "mainstream_trust": 0.5},
        prior_beliefs={"political_lean": political_lean},
        identity_salience={},
        content_sensitivities=ContentSensitivities(),
        persona_anchors="- example anchor",
        natural_language_description="A test persona description.",
    )


def _valid_response(**overrides) -> dict:
    response = {
        "immediate_reaction": {
            "primary_emotion": "engagement",
            "secondary_emotion": "interest",
            "intensity": 6,
            "gut_response": "It held my attention from the first scene.",
        },
        "overall_assessment": {
            "valence": "mixed",
            "confidence": 6,
            "would_watch_full_film": True,
            "would_recommend_to_friend": False,
            "expected_enjoyment_if_watched": 6,
        },
        "element_reactions": [],
        "character_reactions": [],
        "friction_experienced": [],
        "core_reason": "The premise is interesting but execution is uneven.",
        "demographic_lens": "As a 35-year-old urban woman this resonates somewhat.",
    }
    response.update(overrides)
    return response


def test_collapse_phrase_detection() -> None:
    response = _valid_response(
        core_reason="I can appreciate the craft even if it isn't to my taste.",
    )
    is_valid, reason = validate_response(response, _make_agent())
    assert not is_valid
    assert "Collapse" in reason


def test_valid_response_passes() -> None:
    is_valid, reason = validate_response(_valid_response(), _make_agent())
    assert is_valid, reason
    assert reason == "ok"


def test_internal_consistency_check() -> None:
    response = _valid_response(overall_assessment={
        "valence": "negative",
        "confidence": 6,
        "would_watch_full_film": True,
        "expected_enjoyment_if_watched": 8,
        "would_recommend_to_friend": False,
    })
    is_valid, reason = validate_response(response, _make_agent())
    assert not is_valid
    assert "Inconsistency" in reason


def test_demographic_lens_length() -> None:
    empty = _valid_response(demographic_lens="")
    short = _valid_response(demographic_lens="too short")
    adequate = _valid_response(
        demographic_lens="As a midwestern parent this content reads as gratuitous."
    )

    agent = _make_agent()
    assert validate_response(empty, agent)[0] is False
    assert validate_response(short, agent)[0] is False
    assert validate_response(adequate, agent)[0] is True
