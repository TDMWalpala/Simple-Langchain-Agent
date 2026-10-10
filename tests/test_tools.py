"""Tests for :mod:`simple_agent.tools` — search and weather tools.

All HTTP is mocked via the ``patch_weather`` fixture; nothing here touches
the network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import requests
from langchain_tavily import TavilySearch

from simple_agent.tools import MAX_SEARCH_RESULTS, get_search_tool, get_weather

#: Fixture from conftest.py: ``patch_weather(payload, exc=None) -> recorded calls``.
Patcher = Callable[..., dict[str, Any]]

# --- search tool -----------------------------------------------------------


def test_search_tool_returns_tavily_with_default_limit(fake_env: dict[str, str]) -> None:
    tool = get_search_tool()
    assert isinstance(tool, TavilySearch)
    assert tool.max_results == MAX_SEARCH_RESULTS


def test_search_tool_accepts_custom_limit(fake_env: dict[str, str]) -> None:
    assert get_search_tool(max_results=5).max_results == 5


def test_search_tool_fails_fast_without_key(
    monkeypatch: pytest.MonkeyPatch, no_dotenv: None
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        get_search_tool()


# --- weather tool: happy path ----------------------------------------------

WEATHER_OK = {
    "success": True,
    "location": {"name": "Paris"},
    "current": {
        "weather_descriptions": ["Sunny"],
        "temperature": 21,
        "feelslike": 20,
        "wind": 12,
    },
}


def test_weather_reports_conditions(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    patch_weather(WEATHER_OK)
    result = get_weather.invoke({"city": "Paris"})
    assert "Paris" in result
    assert "Sunny" in result
    assert "21°C" in result
    assert "20°C" in result
    assert "12 km/h" in result


def test_weather_sends_expected_request(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    calls = patch_weather(WEATHER_OK)
    get_weather.invoke({"city": "Paris"})
    assert calls["url"] == "https://api.weatherstack.com/current"
    params = calls["kwargs"]["params"]
    assert params["query"] == "Paris"
    assert params["access_key"] == fake_env["WEATHER_API_KEY"]
    assert params["units"] == "m"  # metric: °C and km/h
    assert calls["kwargs"]["timeout"] == 10


def test_weather_multiple_descriptions_are_joined(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    payload = {**WEATHER_OK, "current": {**WEATHER_OK["current"], "weather_descriptions": ["Partly", "Cloudy"]}}
    patch_weather(payload)
    assert "Partly, Cloudy" in get_weather.invoke({"city": "Paris"})


def test_weather_falls_back_to_city_when_location_missing(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    payload = {**WEATHER_OK}
    payload.pop("location")
    patch_weather(payload)
    assert "Weather in Paris:" in get_weather.invoke({"city": "Paris"})


# --- weather tool: failure paths -------------------------------------------


def test_weather_handles_http_200_error_payload(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    """WeatherStack returns 200 + ``{"success": false, ...}`` for API errors."""
    patch_weather({"success": False, "error": {"code": 101, "info": "Invalid API key."}})
    result = get_weather.invoke({"city": "Paris"})
    assert result == "Weather lookup failed for Paris: Invalid API key."


def test_weather_handles_error_payload_without_info(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    patch_weather({"success": False})
    assert "unknown error" in get_weather.invoke({"city": "Paris"})


def test_weather_handles_missing_current_block(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    patch_weather({"success": True})
    result = get_weather.invoke({"city": "Atlantis"})
    assert result == "No current weather data available for Atlantis."


def test_weather_never_emits_empty_description(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    """Regression: empty weather_descriptions used to yield '...is .'"""
    payload = {**WEATHER_OK, "current": {**WEATHER_OK["current"], "weather_descriptions": []}}
    patch_weather(payload)
    result = get_weather.invoke({"city": "Paris"})
    assert "unknown conditions" in result
    assert "is ." not in result


def test_weather_propagates_http_errors(
    fake_env: dict[str, str], patch_weather: Patcher
) -> None:
    """Real HTTP errors (500, timeout-ish statuses) should raise, not be swallowed."""
    patch_weather({}, status_error=requests.HTTPError("503 Server Error"))
    with pytest.raises(requests.HTTPError):
        get_weather.invoke({"city": "Paris"})
