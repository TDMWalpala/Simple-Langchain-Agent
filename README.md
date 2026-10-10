# Simple LangChain Agent

A small, beginner-friendly LangChain agent that answers questions with a Groq
language model and a Tavily web-search tool. The project is organized as a
reusable `src/` package plus an educational Jupyter notebook.

---

## 1. Project description

This project builds a **tool-calling agent**:

1. You ask a question in plain language.
2. The agent asks the LLM whether a tool is needed.
3. If the question needs fresh information, the agent calls **Tavily Search**;
   if it needs live weather, it calls the **weather** tool.
4. The tool results are passed back to the LLM.
5. The LLM writes a final answer.

Everything is intentionally kept small: no databases, no web framework, no
vector store. It is a *first LangChain agent*, not a production system.

## 2. Features

- Modern LangChain API: `create_agent` (no `AgentExecutor`, no
  `create_react_agent`, no `langchainhub` prompts).
- Groq LLM (`openai/gpt-oss-120b`) with `temperature=0` for deterministic answers.
- Tavily search tool for current information.
- Weather tool (WeatherStack) for live conditions, with error handling for
  the API's "HTTP 200 + error payload" failure style.
- A system prompt that tells the agent when to use tools and how to handle
  failures.
- Centralized configuration with validation and clear error messages.
- Reusable `src/simple_agent` package used by the notebook and by `app.py`.
- Command-line entry point: `uv run app.py "your question"`.
- Offline pytest suite (`uv run pytest`): tools, config, model, agent, CLI.
- Educational notebook covering setup, tools, the agent loop, message
  inspection, debugging, experimentation, and error handling.

## 3. Architecture

```text
User
 ↓
LangChain Agent          (create_agent)
 ↓
Groq LLM                 (openai/gpt-oss-120b)
 ↓
Tool decision             (LLM decides: answer, search, or weather)
 ↓
Tavily Search / Weather   (only when fresh information is needed)
 ↓
Tool result
 ↓
LLM                       (reads the tool result)
 ↓
Final answer
```

Each component in beginner-friendly terms:

| Component        | What it does                                                        |
| ---------------- | ------------------------------------------------------------------- |
| **User**         | You type a question into the notebook.                              |
| **Agent**        | The LangChain loop that coordinates the model and its tools.        |
| **LLM**          | The "brain" (Groq-hosted `openai/gpt-oss-120b`) that decides what to do next. |
| **Tool decision**| The model either answers directly or requests a tool call.          |
| **Tavily**       | A search API that returns fresh web results.                        |
| **Tool result**  | The search results, injected back into the conversation as a message. |
| **Final answer** | The model's grounded response, now informed by the search results.  |

## 4. Requirements

- [uv](https://docs.astral.sh/uv/) (dependency and environment manager)
- Python 3.13+ (uv installs it automatically if needed)
- A [Groq API key](https://console.groq.com/keys)
- A [Tavily API key](https://app.tavily.com/home)

## 5. Installation

```bash
git clone <your-repo-url>
cd simple-langchain-agent

# Create/refresh the virtual environment and install all dependencies
uv sync
```

`uv sync` reads `pyproject.toml`, resolves the dependencies, and writes
`uv.lock`. Never edit `uv.lock` by hand.

## 6. Environment variables

Copy the example file and add your keys:

```bash
cp .env.example .env
```

`.env`:

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
WEATHER_API_KEY=your_weatherstack_api_key
```

| Variable          | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `GROQ_API_KEY`    | Authenticates requests to the Groq API.          |
| `TAVILY_API_KEY`  | Authenticates requests to the Tavily API.        |
| `WEATHER_API_KEY` | Authenticates requests to the WeatherStack API.  |

`.env` is listed in `.gitignore` and must never be committed. The application
loads and validates these values in `src/simple_agent/config.py`; API keys are
never hard-coded in source code or notebooks.

## 7. Running the notebook

```bash
uv run jupyter notebook
```

Then open `notebooks/simple_agent.ipynb` and run the cells from top to bottom.

The notebook is the main learning interface. Section order:

1. Project overview
2. Environment setup
3. Load configuration
4. Test Tavily
5. Initialize the LLM
6. Create the agent
7. Run the agent
8. Inspect agent execution (message flow + debug logging)
9. Experimentation
10. Error handling
11. Architecture explanation

## 8. Running the tests

```bash
uv run pytest
```

The suite lives in `tests/` and is fully offline: HTTP is mocked and fake API
keys are injected, so no real keys (or network) are needed to run it.

| File                 | What it covers                                              |
| -------------------- | ----------------------------------------------------------- |
| `test_config.py`     | Env loading and fail-fast key validation.                   |
| `test_tools.py`      | Search + weather tools, including WeatherStack's error style. |
| `test_model.py`      | Groq model factory configuration.                           |
| `test_agent.py`      | Agent wiring: model, tools, and system prompt.              |
| `test_app.py`        | CLI entry point (question handling, recursion limit).       |

## 9. Project structure

```text
simple-langchain-agent/
│
├── notebooks/
│   └── simple_agent.ipynb      # Educational walkthrough (start here)
│
├── tests/                      # Pytest suite (offline, HTTP mocked)
│   ├── conftest.py             # Shared fixtures (fake keys, mocked HTTP)
│   ├── test_agent.py
│   ├── test_app.py
│   ├── test_config.py
│   ├── test_model.py
│   └── test_tools.py
│
├── src/
│   └── simple_agent/
│       ├── __init__.py         # Public API of the package
│       ├── config.py           # Loads and validates environment variables
│       ├── model.py            # get_llm() -> configured ChatGroq model
│       ├── tools.py            # get_search_tool() + get_weather() tools
│       └── agent.py            # create_agent() -> LangChain agent
│
├── app.py                      # CLI entry point: uv run app.py "question"
├── .env                        # Your keys (git-ignored)
├── .env.example                # Template with placeholder keys
├── .gitignore
├── pyproject.toml              # Dependencies managed by uv
├── README.md
└── uv.lock                     # Lock file (generated, do not edit)
```

You can also run the agent from the command line without Jupyter:

```bash
uv run app.py "What is the current weather in Paris?"
```

## 10. How the agent works

```python
from simple_agent import create_agent

agent = create_agent()                     # LLM + tools are wired together
answer = agent.invoke({
    "messages": [{"role": "user", "content": "What is the capital of Austria?"}]
})
print(answer["messages"][-1].content)
```

Under the hood:

1. `get_llm()` returns a `ChatGroq` model configured with `openai/gpt-oss-120b`,
   `temperature=0`, and `max_retries=3` (Groq rate-limits aggressively).
2. `get_search_tool()` returns a Tavily search tool limited to `max_results=2`;
   `get_weather()` returns the WeatherStack lookup tool.
3. `create_agent(model=llm, tools=tools, system_prompt=...)` builds a LangGraph graph that:
   - sends the conversation to the LLM,
   - executes a tool if the LLM requests one,
   - sends the tool result back to the LLM,
   - repeats until the LLM produces a final answer.

Legacy approaches (`hub.pull("hwchase17/react")`, `create_react_agent`,
`AgentExecutor`) are no longer needed: tool calling is now native to the model,
so the agent loop lives in a LangGraph graph instead of a hand-written ReAct
prompt.

## 11. Example questions

| Question                                            | Expected behaviour                              |
| --------------------------------------------------- | ----------------------------------------------- |
| `What is the capital of Austria?`                   | Answered directly, **no** tool call.            |
| `What is the current population of Austria?`        | Uses **Tavily** for fresh data.                 |
| `What is the current weather in Paris?`             | Uses the **weather** tool.                      |
| `Search for the latest news about Austria and summarize it in one sentence.` | Uses Tavily, then summarizes. |
| `What is the meaning of life?`                      | Answered conversationally, no tool needed.      |

## 12. Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `ValueError: Missing required environment variable(s): ...` | Create `.env` from `.env.example` and fill in the keys. |
| `401 Unauthorized` from Groq | Check `GROQ_API_KEY` and that the key is active. |
| Tavily raises an authentication/`401` error | Check `TAVILY_API_KEY`. |
| `ModuleNotFoundError: No module named 'simple_agent'` | Run `uv sync` and start Jupyter with `uv run jupyter notebook`. |
| SSL/certificate errors (corporate proxy, Windows) | `config.py` sets `SSL_CERT_FILE` to the `certifi` bundle; keep `certifi` installed. |
| Rate limit errors from Groq | Wait a moment and retry; the free tier has per-minute limits. |
| Stale dependencies | Run `uv sync --upgrade` (never hand-edit `uv.lock`). |

## 13. Future improvements

- Add more tools (calculator, file reading) to see how the agent routes
  between several tools.
- Persist conversation memory with a LangGraph checkpointer.
- Add structured output (`response_format`) for machine-readable answers.
- Evaluate the agent with LangSmith tracing (a pytest suite already exists;
  see `tests/`).
