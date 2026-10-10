"""Groq chat model factory.

The model is created in exactly one place so that every part of the project
(notebook, scripts, future tests) uses the same configuration.
"""

from __future__ import annotations

from langchain_groq import ChatGroq

from .config import load_config

#: Groq model identifier used across the project.
GROQ_MODEL = "openai/gpt-oss-120b"


def get_llm() -> ChatGroq:
    """Create and return the configured Groq chat model.

    The model is ``openai/gpt-oss-120b`` served by Groq. Temperature is set to
    ``0`` so that the model is as deterministic as possible: for a learning
    project (and for tool calling in general) we want reproducible answers and
    reliable tool selection rather than creative variation.

    The API key is read from the ``GROQ_API_KEY`` environment variable by
    ``ChatGroq`` itself; :func:`load_config` makes sure it exists first so the
    failure message is clear instead of an opaque 401 later on.

    Returns:
        ChatGroq: Configured Groq chat model.

    Raises:
        ValueError: If ``GROQ_API_KEY`` is not configured.
    """
    load_config()
    return ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=3)
