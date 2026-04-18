"""Aggregation tests."""

from __future__ import annotations

import pytest

from agents.schemas import (
    AgentReaction,
    AgentResult,
    ImmediateReaction,
    OverallAssessment,
)
from aggregation.aggregate import aggregate


def _make_result(
    agent_id: str,
    valence: str,
    *,
    political_lean: float = 0.0,
    gender: str = "female",
    age: int = 30,
    openness: float = 0.5,
) -> AgentResult:
    return AgentResult(
        agent_id=agent_id,
        reaction=AgentReaction(
            immediate_reaction=ImmediateReaction(
                primary_emotion="engagement",
                secondary_emotion="interest",
                intensity=6,
                gut_response="ok",
            ),
            overall_assessment=OverallAssessment(
                valence=valence,
                confidence=6,
                would_watch_full_film=valence == "positive",
                would_recommend_to_friend=valence == "positive",
                expected_enjoyment_if_watched=7 if valence == "positive" else 3,
            ),
            element_reactions=[],
            character_reactions=[],
            friction_experienced=[],
            core_reason="placeholder",
            demographic_lens="placeholder lens long enough to pass",
        ),
        demographics={"age": age, "gender": gender, "education": "college",
                      "geography": "urban", "region": "northeast",
                      "income_bracket": "middle"},
        psychographics={"openness": openness, "conscientiousness": 0.5,
                        "extraversion": 0.5, "agreeableness": 0.5,
                        "neuroticism": 0.5},
        moral_foundations={"care": 0.5, "fairness": 0.5, "loyalty": 0.5,
                           "authority": 0.5, "purity": 0.5, "liberty": 0.5},
        prior_beliefs={"political_lean": political_lean},
    )


def test_aggregate_empty_results() -> None:
    out = aggregate([])
    assert out == {"overall": {}, "segments": {}, "element_attribution": {}}


def test_aggregate_computes_positive_rate() -> None:
    results = [
        _make_result("a", "positive"),
        _make_result("b", "positive"),
        _make_result("c", "negative"),
    ]
    out = aggregate(results)
    assert out["overall"]["n"] == 3
    assert out["overall"]["positive_rate"] == pytest.approx(2 / 3)
    assert out["overall"]["negative_rate"] == pytest.approx(1 / 3)


def test_segment_filtering() -> None:
    results = [
        _make_result("p1", "positive", political_lean=-0.6,
                     gender="female", age=28, openness=0.8),
        _make_result("p2", "positive", political_lean=-0.5,
                     gender="female", age=30, openness=0.75),
        _make_result("c1", "negative", political_lean=0.7,
                     gender="male", age=55, openness=0.3),
    ]
    out = aggregate(results)
    segs = out["segments"]
    assert "progressive_women" in segs
    assert segs["progressive_women"]["n"] == 2
    assert "conservative_men" in segs
    assert segs["conservative_men"]["n"] == 1
    # The conservative_men segment must not include the progressive women
    assert segs["progressive_women"]["positive_rate"] == 1.0
    assert segs["conservative_men"]["negative_rate"] == 1.0


