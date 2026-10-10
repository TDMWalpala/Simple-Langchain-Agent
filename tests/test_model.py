"""Tests for :mod:`simple_agent.model` — the Groq LLM factory."""

from __future__ import annotations

import pytest
from langchain_groq import ChatGroq

from simple_agent.model import GROQ_MODEL, get_llm


def test_get_llm_returns_configured_chatgroq(fake_env: dict[str, str]) -> None:
    llm = get_llm()
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == GROQ_MODEL
    # We ask for temperature=0; langchain-groq coerces it to 1e-8 because the
    # Groq API rejects exactly 0. Either way: maximally deterministic.
    assert llm.temperature == pytest.approx(0, abs=1e-7)
    assert llm.max_retries == 3  # Groq rate-limits aggressively


def test_get_llm_fails_fast_without_key(
    monkeypatch: pytest.MonkeyPatch, no_dotenv: None
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()
