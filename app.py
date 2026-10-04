# %%
import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent




# %%
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# %%
search_tool = TavilySearchResults(
    max_results=2,)

# %%
result = search_tool.invoke({"query": "What is the capital of France?"})
result

# %%
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)

# %%
response = llm.invoke("What is the capital of Austria?")
response

# %%
tools = [search_tool]

# %%
agent = create_agent(
    model=llm,
    tools=tools,
)

# %%
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the current population of germany 2026?"
            }
        ]
    }
)

print(result["messages"][-1].content)


