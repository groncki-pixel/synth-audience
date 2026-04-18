"""
Shared pytest fixtures.

The sampler reads lookup tables from ``data/processed/`` at module import.
The preprocessor's hardcoded tables (IPIP, MFQ) work without any raw CSVs,
so we rebuild them once per test session and reload the sampler so the
module-level constants reflect the new files.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def _build_lookup_tables() -> None:
    from agents import preprocessor, sampler

    preprocessor.build_lookup_tables()
    importlib.reload(sampler)
