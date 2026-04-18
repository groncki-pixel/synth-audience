"""Sampler tests.

These exercise the public sample_agents API plus the deterministic content
sensitivity derivation to confirm extreme inputs produce different
outputs.
"""

from __future__ import annotations

import numpy as np

from agents.sampler import _derive_content_sensitivities, sample_agents
from agents.schemas import DemographicTarget


def _target() -> DemographicTarget:
    return DemographicTarget(
        age_min=30,
        age_max=40,
        gender="female",
        education="college",
        geography="urban",
        region="northeast",
        income_bracket="middle",
    )


def test_sample_single_agent_returns_valid_profile() -> None:
    agents = sample_agents(_target(), n_agents=1)
    assert len(agents) == 1
    a = agents[0]

    assert a.agent_id
    assert a.demographics["age"] == 30 or 30 <= a.demographics["age"] <= 40
    assert a.psychographics, "Big Five missing"
    for trait in ["openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"]:
        v = a.psychographics[trait]
        assert 0.0 <= v <= 1.0, f"{trait}={v} not in [0,1]"

    for trait in ["care", "fairness", "loyalty", "authority", "purity", "liberty"]:
        v = a.moral_foundations[trait]
        assert 0.0 <= v <= 1.0, f"MFQ {trait}={v} not in [0,1]"

    pol = a.prior_beliefs["political_lean"]
    assert -1.0 <= pol <= 1.0


def test_sample_multiple_agents_produces_variance() -> None:
    np.random.seed(0)
    agents = sample_agents(_target(), n_agents=20)
    openness = [a.psychographics["openness"] for a in agents]
    assert float(np.std(openness)) > 0.0


def test_content_sensitivities_respond_to_inputs() -> None:
    big5 = {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
    }
    high_purity_conservative = {
        "care": 0.5, "fairness": 0.5, "loyalty": 0.7,
        "authority": 0.85, "purity": 0.95, "liberty": 0.4,
    }
    low_purity_progressive = {
        "care": 0.85, "fairness": 0.85, "loyalty": 0.3,
        "authority": 0.2, "purity": 0.05, "liberty": 0.8,
    }

    cons = _derive_content_sensitivities(big5, high_purity_conservative, 0.8)
    prog = _derive_content_sensitivities(big5, low_purity_progressive, -0.8)

    assert cons.sex_work_depictions != prog.sex_work_depictions
    assert cons.sex_work_depictions == "high_salience"
    assert prog.sex_work_depictions == "low_salience"
