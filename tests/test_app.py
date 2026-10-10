"""Tests for the ``app.py`` CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import app
import pytest


def _fake_agent(answer: str = "Vienna.") -> MagicMock:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [MagicMock(content=answer)]}
    return agent


def test_main_asks_the_given_question(capsys: pytest.CaptureFixture[str]) -> None:
    agent = _fake_agent("It is sunny.")
    with patch.object(app, "create_agent", return_value=agent):
        exit_code = app.main(["What is the weather in Paris?"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "It is sunny."

    inputs = agent.invoke.call_args.args[0]
    assert inputs["messages"][0] == {"role": "user", "content": "What is the weather in Paris?"}


def test_main_uses_default_question_when_none_given() -> None:
    agent = _fake_agent()
    with patch.object(app, "create_agent", return_value=agent):
        app.main([])

    inputs = agent.invoke.call_args.args[0]
    assert inputs["messages"][0]["content"] == app.DEFAULT_QUESTION


def test_main_caps_recursion_limit() -> None:
    """Bound the agent loop so a tool-calling loop can't run forever."""
    agent = _fake_agent()
    with patch.object(app, "create_agent", return_value=agent):
        app.main([])

    config = agent.invoke.call_args.kwargs["config"]
    assert config["recursion_limit"] == 15
