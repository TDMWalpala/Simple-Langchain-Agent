"""Tools available to the agent.

A *tool* is a function the LLM is allowed to call while answering a question.
This project has two tools: Tavily web search — which lets the agent answer
questions about **current** information that the model was not trained on —
and a WeatherStack lookup for current weather conditions.
"""

from __future__ import annotations

import os

import requests
from langchain.tools import tool
from langchain_tavily import TavilySearch

from .config import load_config

#: Number of search results the tool returns (keeps prompts small and cheap).
MAX_SEARCH_RESULTS = 2


def get_search_tool(max_results: int = MAX_SEARCH_RESULTS) -> TavilySearch:
    """Create and return the Tavily web-search tool.

    `Tavily <https://www.tavily.com>`_ is a search API built for LLM agents:
    it returns clean, ranked snippets instead of raw HTML, so the model can
    read them directly.

    Why does the agent need it? A language model only knows what it saw during
    training, so it cannot answer questions like *"What is the current
    population of Austria?"* reliably. When the agent decides the question is
    about recent or factual web information, it calls this tool, feeds the
    results back to the model, and the model writes a grounded answer.

    The API key is read from the ``TAVILY_API_KEY`` environment variable by the
    tool itself; this function only validates that it exists. Keys are never
    hard-coded here.

    Args:
        max_results: How many search results to return. Keeping this small
            (2 by default) limits how much text is sent back to the model.

    Returns:
        TavilySearch: A ready-to-use search tool.

    Raises:
        ValueError: If ``TAVILY_API_KEY`` is not configured.
    """
    load_config()
    return TavilySearch(max_results=max_results)


@tool
def get_weather(city: str) -> str:
    """Get current weather conditions for a city.

    Args:
        city: Name of the city to look up, e.g. ``"Paris"``.

    Returns:
        A one-line summary with temperature in °C, apparent ("feels like")
        temperature, wind speed in km/h, and a plain-text sky description —
        or a clear error message if the lookup fails.
    """
    load_config()
    response = requests.get(
        "https://api.weatherstack.com/current",
        params={
            "access_key": os.environ["WEATHER_API_KEY"],
            "query": city,
            "units": "m",  # metric: °C and km/h
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    # WeatherStack reports failures as HTTP 200 with an "error" payload,
    # so raise_for_status() alone would not catch them.
    if not data.get("success", True):
        info = data.get("error", {}).get("info", "unknown error")
        return f"Weather lookup failed for {city}: {info}"

    current = data.get("current", {})
    if not current:
        return f"No current weather data available for {city}."

    description = ", ".join(current.get("weather_descriptions", [])) or "unknown conditions"
    location = data.get("location", {}).get("name", city)
    return (
        f"Weather in {location}: {description}. "
        f"Temperature {current.get('temperature')}°C "
        f"(feels like {current.get('feelslike')}°C), "
        f"wind {current.get('wind')} km/h."
    )
