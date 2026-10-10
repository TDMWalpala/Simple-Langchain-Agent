"""Agent assembly.

The agent is the loop that ties everything together:

1. the Groq LLM (:func:`simple_agent.model.get_llm`),
2. the Tavily search tool (:func:`simple_agent.tools.get_search_tool`),
3. the weather tool (:func:`simple_agent.tools.get_weather`),
4. LangChain's modern :func:`langchain.agents.create_agent`.

``create_agent`` builds a LangGraph graph that repeatedly asks the model what
to do next: answer the user, or call a tool. This replaces the legacy
``create_react_agent`` + ``AgentExecutor`` + ``langchainhub`` prompt stack.
"""

from __future__ import annotations

from typing import Any

# Imported under an alias because this module exports its own `create_agent`.
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.runnables import Runnable

from .model import get_llm
from .tools import get_search_tool, get_weather

#: Shapes the agent's tone and tool-use policy. Without a system prompt the
#: model gets no guidance on when to search or how to handle tool failures.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools for factual, current, or "
    "location-specific questions rather than relying on your own knowledge. "
    "If a tool fails, say so plainly instead of guessing."
)


def create_agent() -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build the LangChain tool-calling agent.

    The function wires the project's building blocks together:

    * it loads the configured Groq LLM,
    * it loads the available tools (Tavily search + weather lookup),
    * it passes both to :func:`langchain.agents.create_agent`.

    How the agent works at runtime: the incoming question is sent to the LLM.
    If the LLM replies with a *tool call*, the agent executes the tool and
    sends the result back to the LLM; otherwise the reply is the final answer.
    The loop continues until the LLM stops asking for tools.

    Returns:
        Runnable: A runnable agent. Invoke it with
        ``{"messages": [{"role": "user", "content": "..."}]}`` and read the
        final answer from ``result["messages"][-1].content``.

    Raises:
        ValueError: If a required API key is missing.
    """
    llm = get_llm()
    tools = [get_search_tool(), get_weather]

    return create_langchain_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
