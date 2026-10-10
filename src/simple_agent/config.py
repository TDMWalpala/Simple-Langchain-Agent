"""Configuration loading for the simple LangChain agent.

This module is the single place where environment variables are loaded and
validated. It is intentionally small:

* it loads ``.env`` with :mod:`dotenv`,
* it points Python at the ``certifi`` CA bundle (helps on Windows / behind
  corporate proxies),
* it validates that every required API key is present, and raises a clear
  error when one is missing.

API keys are **never** hard-coded; they always come from the environment.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import certifi
from dotenv import load_dotenv

#: Environment variables the application cannot run without.
REQUIRED_KEYS: tuple[str, ...] = ("GROQ_API_KEY", "TAVILY_API_KEY", "WEATHER_API_KEY")


def missing_keys(keys: Sequence[str] = REQUIRED_KEYS) -> list[str]:
    """Return the subset of ``keys`` that is not set (or is empty).

    Args:
        keys: Environment variable names to check.

    Returns:
        The names of the variables that are missing or empty.
    """
    return [key for key in keys if not os.getenv(key)]


def require_keys(keys: Sequence[str]) -> None:
    """Raise ``ValueError`` if any of ``keys`` is missing from the environment.

    Args:
        keys: Environment variable names that must be present.

    Raises:
        ValueError: If at least one variable is missing or empty. The message
            lists every missing variable and points to ``.env.example``.
    """
    missing = missing_keys(keys)
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your API keys."
        )


def load_config() -> None:
    """Load ``.env`` and validate the required API keys.

    Calling this more than once is safe: the underlying ``load_dotenv`` call
    is idempotent and never overwrites variables that are already set.

    Raises:
        ValueError: If ``GROQ_API_KEY``, ``TAVILY_API_KEY`` or
            ``WEATHER_API_KEY`` is not configured.
    """
    # Use certifi's CA bundle for HTTPS requests (Groq, Tavily, ...).
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    load_dotenv()
    require_keys(REQUIRED_KEYS)
