"""Shared fixtures.

Tests never touch the network and never need real API keys: HTTP is mocked
and fake keys are injected into the environment.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

#: Fake keys so config validation passes without a real ``.env`` file.
FAKE_KEYS: dict[str, str] = {
    "GROQ_API_KEY": "gsk_test_key",
    "TAVILY_API_KEY": "tvly_test_key",
    "WEATHER_API_KEY": "test_weather_key",
}


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set fake API keys for the duration of a test."""
    for name, value in FAKE_KEYS.items():
        monkeypatch.setenv(name, value)
    return FAKE_KEYS


@pytest.fixture
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``load_config()`` ignore any real ``.env`` on disk.

    Without this, a developer's real ``.env`` would silently satisfy the
    key validation and hide failures in tests that assert on missing keys.
    """
    monkeypatch.setattr("simple_agent.config.load_dotenv", lambda *a, **k: None)


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response` used by tool tests."""

    def __init__(self, payload: dict, status_error: Exception | None = None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def patch_weather(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict, Exception | None], dict]:
    """Replace ``requests.get`` for the weather tool and record the call.

    Usage::

        calls = patch_weather({"success": True, "current": {...}})
        get_weather.invoke({"city": "Paris"})
        assert calls["kwargs"]["params"]["query"] == "Paris"
    """
    calls: dict = {}

    def _patch(payload: dict, status_error: Exception | None = None) -> dict:
        response = FakeResponse(payload, status_error)

        def fake_get(url: str, **kwargs: object) -> FakeResponse:
            calls["url"] = url
            calls["kwargs"] = kwargs
            return response

        monkeypatch.setattr("simple_agent.tools.requests.get", fake_get)
        return calls

    return _patch
