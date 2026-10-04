# %%
import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain.tools import tool


# %%
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# %%
search_tool = TavilySearchResults(
    max_results=2,)

# %%
@tool
def weather_tool(city: str) -> str:
    """Get the current weather for a city."""
    response = requests.get(
        "https://api.weatherstack.com/current",
        params={"access_key": WEATHER_API_KEY, "query": city},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if "current" not in data:
        return f"Could not retrieve weather data for {city}."

    descriptions = data["current"].get("weather_descriptions", [])
    return f"The weather in {city} is {', '.join(descriptions)}."

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
tools = [search_tool, weather_tool]

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
                "content": "find the current weather in Paris and the capital of Austria",
            }
        ]
    }
)

print(result["messages"][-1].content)


