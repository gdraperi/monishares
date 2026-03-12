
import os
from pyexpat import model
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import csv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

from IPython.display import Image, display
from exa_py import Exa
from datetime import datetime, timedelta

load_dotenv()
exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

""""
import polars as pl

def load_multi_column_data(state: GraphState):
    # Load the CSV using Polars
    df = pl.read_csv("data.csv")
    
    # Select only the columns you need (e.g., 'id', 'price', 'stock_level')
    selected_df = df.select([
        pl.col("product_id"),
        pl.col("price"),
        pl.col("stock_level")
    ])
    
    # Convert to a list of dictionaries for the State
    data_as_dicts = selected_df.to_dicts()
    
    return {"inventory_data": data_as_dicts}
    """

"""
def assistant_node(state: GraphState):
    # Access the columns we loaded earlier
    context = state["inventory_data"]
    
    # Construct a string for the LLM to read
    formatted_context = "\n".join([str(row) for row in context])
    
    prompt = f"System: Use this data to answer the user:\n{formatted_context}"
    # (Send to LLM here...)
    return {"messages": ["I've analyzed the columns!"]}"""





class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]




def load_shares(path: str):
        """Load shares from a CSV file and print their names."""
        with open(path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(row['Libellé'])


def search_and_contents(content: str):
    """Search for content and print the results through EXA API."""
    results = exa.search_and_contents(content,
    type="deep",
    num_results=10,
    highlights={"max_characters": 4000}
    )

    for result in results.results:
        print(result.title, result.url)

@tool
def search_and_contents_tool(content: str):
    """Search for content and print the results through EXA API.
    
    Arg 1: content (str): The query to search for.
    """
    # 1. Define the 'Daily' window
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    #print(f"Searching for news related to: {content} from {yesterday} onwards.")

    results = exa.search_and_contents(content,
    type="neural",                      # Optimized for natural language queries
    start_published_date=yesterday,
    # category="news",                  # Pro-tip: Omit this for faster, broader news coverage
    num_results=5,
    max_age_hours=24,                   # Modern way to ensure "Live" data within a 24h window
    contents={
        "highlights": {
            "num_sentences": 3,         # Precisely what you asked for
            "highlights_per_url": 1     # Ensures variety across different news sources
        },
        "text": False                   # Drop the full text to save tokens/memory
    }
    )

    return results

tools = [search_and_contents_tool]

model= ChatOpenAI(model = "gpt-4o").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
      # Get current date in a readable format
    today = datetime.now().strftime("%B %d, %y")

    prompt = f"""
### ROLE
You are a Senior Quantitative Equity Analyst specializing in the CAC 40 index. Your goal is to provide a real-time investment briefing for today FOCUSED ON CAC40, {today}.

### ANALYSIS CRITERIA
1. MACRO VIEW: Briefly summarize the overall sentiment of the Paris CAC40 Bourse {today}.
2. TOP MOVERS: Identify which stocks are leading and lagging. 
3. INVESTMENT PICKS: Recommend 2 'Buy' candidates based on current technical levels or news catalysts (earnings, M&A, dividends).
4. REASONING: For every suggestion, cite a specific data point found via Exa.

### OUTPUT STRUCTURE
- **Market Pulse**: 1-sentence summary.
- **Top Values to Follow**: A bulleted list of 3-5 tickers with 1-sentence context each.
- **Investment Spotlight**: Deep dive into 2 specific stocks with 'Interesting for Investing' labels.
- **Risk Warning**: A brief note on volatility or upcoming ECB/Macro events.
### TOOL INSTRUCTIONS: Remember to use the Exa API to back up your analysis with real-time data and insights.
When calling the search tool, do not use keywords like "CAC 40 stocks today." Instead, use "Social Proof" or "Neural Completion" style queries. 
- Example: "Here is a technical analysis of the top-performing CAC 40 stocks as of {today}:"

### Once finished with your analysis, end with "END LLM" to signal completion.
"""

    
    system_prompt = SystemMessage(content=prompt)
    user_input = input("\nDo you want to continue?")
    print(f"\n USER: {user_input}")
    user_message = HumanMessage(content=user_input)
    if "END LLM" in user_input:
        return {"messages": [user_message]}
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState): 
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        print(message)
        if("end llm" in message.content.lower()):
            return "end"

    return "continue"

graph= StateGraph(AgentState)

graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent","tools")

graph.add_conditional_edges(

    "tools",
    should_continue,
    {
        "continue":"agent",
        "end": END,
    },
)

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()



if  __name__ == "__main__":
    print("Monishares!")

    
    #share_path="shares/Export_portefeuille_simple_011920211816.csv"
    #load_shares(share_path)
    #search_and_contents("CAC40 stocks performance today.")
    inputs = {"messages": [("user", "")]}
    print_stream(app.stream(inputs,stream_mode="values"))