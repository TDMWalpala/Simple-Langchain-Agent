"""simple_agent — a small, beginner-friendly LangChain agent.

Public API::

    from simple_agent import load_config, get_llm, get_search_tool, create_agent

Typical use::

    agent = create_agent()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the capital of Austria?"}]}
    )
    print(response["messages"][-1].content)
"""

from .agent import create_agent
from .config import load_config, missing_keys, require_keys
from .model import get_llm
from .tools import get_search_tool, get_weather

__all__ = [
    "create_agent",
    "get_llm",
    "get_search_tool",
    "get_weather",
    "load_config",
    "missing_keys",
    "require_keys",
]
