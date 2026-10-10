"""Tests for :mod:`simple_agent.agent` — agent assembly.

``create_agent()`` is tested by patching its collaborators, so no LLM call
and no network access happens.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

import simple_agent.agent as agent_module
from simple_agent.agent import SYSTEM_PROMPT, create_agent


def test_create_agent_wires_llm_tools_and_prompt() -> None:
    fake_llm = MagicMock(name="llm")
    fake_agent = MagicMock(name="agent")

    with (
        patch.object(agent_module, "get_llm", return_value=fake_llm),
        patch.object(agent_module, "create_langchain_agent", return_value=fake_agent) as factory,
    ):
        result = create_agent()

    assert result is fake_agent
    kwargs = factory.call_args.kwargs
    assert kwargs["model"] is fake_llm
    assert kwargs["system_prompt"] == SYSTEM_PROMPT

    tools = kwargs["tools"]
    assert len(tools) == 2
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {t.name for t in tools} == {"tavily_search", "get_weather"}


def test_create_agent_propagates_missing_key_error(
    monkeypatch: pytest.MonkeyPatch, no_dotenv: None
) -> None:
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "WEATHER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Missing required environment variable"):
        create_agent()


def test_system_prompt_states_tool_policy() -> None:
    """The prompt is the agent's contract: it must say when/how to use tools."""
    lowered = SYSTEM_PROMPT.lower()
    assert "tool" in lowered
    assert "fail" in lowered  # tells the model how to handle tool errors
