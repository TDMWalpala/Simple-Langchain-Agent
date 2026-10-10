"""Command-line entry point for the simple LangChain agent.

The real logic lives in the ``src/simple_agent`` package; this script just
wires it to the terminal::

    uv run app.py "What is the current weather in Paris?"
    uv run app.py            # uses the default demo question
"""

from __future__ import annotations

import argparse
import sys

from simple_agent import create_agent

DEFAULT_QUESTION = "Find the current weather in Paris and the capital of Austria."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the LangChain agent a question.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Question to ask (defaults to a demo question).",
    )
    args = parser.parse_args(argv)

    agent = create_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.question}]},
        config={"recursion_limit": 15},
    )
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
