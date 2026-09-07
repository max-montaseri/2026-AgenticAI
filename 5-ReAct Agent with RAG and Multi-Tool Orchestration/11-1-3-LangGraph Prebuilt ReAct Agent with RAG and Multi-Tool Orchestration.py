# Using prebuilt components ( LangGraph ReAct agent) for rapid development

# The LangGraph library provides prebuilt agent components, such as the ReAct agent, that
# encapsulate much of orchestration logic.
# simplify the agent by switching to a prebuilt approach.

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional
from dotenv import load_dotenv, find_dotenv
import random


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

# Step 1: IMPORTING THE REMAININGSTEPS UTILITY for using the LangGraph ReAct agent
from langgraph.managed.is_last_step import RemainingSteps

from langchain_core.tools import tool

# The function from langchain.agents import create_agent is the modern, official replacement for 
# from langgraph.prebuilt import create_react_agent.
from langchain.agents import create_agent

#region Settings
def load_project_dotenv():
    # load the environment variables from the .env 
    load_dotenv()
def get_llm(llm_model: Optional[str]= None, 
            use_responses_api: Optional[bool] = None, 
            use_previous_response_id: Optional[bool] = None):
    
    if (llm_model is None):
        if (use_responses_api is not None) and (use_previous_response_id is not None):
            return "please send the llm_model's value!"  
        
        if (use_responses_api is None) and (use_previous_response_id is None):
            # return Cheapest LLM
            openai_api_key = get_env_api_key ("OpenAI")
            llm_model = get_llm_model("GPT-Cheapest")
            llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)

    elif (llm_model is not None):
        if (use_responses_api is None) and (use_previous_response_id is None):
            llm = ChatOpenAI(llm_model)
        
        if (use_responses_api is not None) and (use_previous_response_id is None):
            llm = ChatOpenAI(model=llm_model, use_responses_api=(True if use_responses_api else False))
        
        if (use_responses_api is not None) and (use_previous_response_id is not None):
            llm = ChatOpenAI(llm_model,
                            use_responses_api=(True if use_responses_api else False),
                            use_previous_response_id=(True if use_previous_response_id else False))
    
    return llm
def get_env_api_key(ai_company):
    ## Get API Key from .env file, based on AI company
    # This automatically searches up the directory tree to find the .env file
    load_dotenv(find_dotenv())
    # load_dotenv()
    match ai_company:
        case "OpenAI": 
            # print(os.environ.get("OPENAI_API_KEY"))
            return os.getenv("OPENAI_API_KEY")
        case "Groq":
            return os.getenv("GROQ_API_KEY")
        case _:  # default / catch-all case
            return "no API Key for this AI company"
def get_llm_model(llm_option):
    # Initialize the model
    match llm_option:
        case "GPT-Cheapest": 
            return "gpt-5-nano"
        case "Groq-Free":
            return "groq"
        case _:  # default / catch-all case
            return llm_option    
        # case _:  # default / catch-all case
        #     return "no llm for this option" 
#endregion
        
# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------

load_project_dotenv()

# -----------------------------------------------------------------------------
# 1. Prepare Travel Info knowledge base (travel information vector store) at startup
# -----------------------------------------------------------------------------

UK_DESTINATIONS = [ # Destination list; we can add more destinations here
    "Cornwall",
    "North_Cornwall",
    "South_Cornwall",
    "West_Cornwall",
]

async def rag_build_vectorstore(destinations: Sequence[str]) -> Chroma: 
    #Function to build the vectorstore and return a reference to the vectorstore client
    """Download WikiVoyage pages and create a Chroma vector store."""
    urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations] #C
    # Disable SSL verification (development only):
    # If you're just experimenting, you can disable SSL verification. Do not do this in production.
    loader = AsyncHtmlLoader(urls, verify_ssl=False)
    print("Downloading destination pages ...")
    docs = await loader.aload() #Load the destination pages asynchronously from the web into a list of documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128) #Split the documents into chunks of 1024 characters with 128 characters of overlap
    chunks = sum([splitter.split_documents([d]) for d in docs], []) 

    print(f"Embedding {len(chunks)} chunks ...") 
    vectordb_client = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings()) #Embed the chunks and store them in the vectorstore
    print("Vector store ready.\n")
    return vectordb_client 


# Singleton pattern (build once)
_travel_info_vectorstore_client: Chroma | None = None #Initialize a cache for the vectorstore client instance as None

def rag_get_travel_info_vectorstore() -> Chroma: 
    #Function to trigger the creation of the vectorstore and return a reference to the cache of its client instance
    global _travel_info_vectorstore_client
    if _travel_info_vectorstore_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set the OPENAI_API_KEY env variable and re-run.")
        _travel_info_vectorstore_client = asyncio.run(rag_build_vectorstore(UK_DESTINATIONS))
    return _travel_info_vectorstore_client #Return the a reference to the cache of the vectorstore client instance

travel_info_vectorstore_client = rag_get_travel_info_vectorstore() #Instantiate the vectorstore client
travel_info_rag_retriever = travel_info_vectorstore_client.as_retriever() #Instantiate the vectorstore retriever

# ----------------------------------------------------------------------------
# 2-1 Define Travel Info tool
# ----------------------------------------------------------------------------

@tool(description="""Search travel information 
about destinations in England.""") #Define the tool using the @tool decorator
def tool_search_travel_info(query: str) -> str: 
    #Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
    """Search embedded WikiVoyage content for information about destinations in England."""
    docs = travel_info_rag_retriever.invoke(query) #Perform a semantic search on the vectorstore and return the top 4 results
    top = docs[:4] if isinstance(docs, list) else docs 
    return "\n---\n".join(d.page_content for d in top) #Joins the top 4 results into a single string

# -----------------------------------------------------------------------------
# 2-2 Define WeatherForecastService (Mock) Tool
# -----------------------------------------------------------------------------

class DictToolWeatherForecast(TypedDict):
    town: str
    weather: Literal["sunny", "foggy", "rainy", "windy"]
    temperature: int

class ToolWeatherForecastService:

    _weather_options = ["sunny", "foggy", "rainy", "windy"]
    _temp_min = 18
    _temp_max = 31

    @classmethod
    def tool_get_forecast(cls, town: str) -> Optional[DictToolWeatherForecast]: 
        #Define the tool_get_forecast method, which returns a DictToolWeatherForecast object
        weather = random.choice(cls._weather_options)
        temperature = random.randint(cls._temp_min, cls._temp_max)
        return DictToolWeatherForecast(town=town, weather=weather, temperature=temperature)

    
@tool(description="Get the weather forecast, given a town name.")
def tool_weather_forecast(town: str) -> dict:
    """Get a mock weather forecast for a given town. Returns a DictToolWeatherForecast object with weather and temperature."""
    forecast = ToolWeatherForecastService.tool_get_forecast(town)
    if forecast is None:
        return {"error": f"No weather data available for '{town}'."}
    return forecast

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [tool_search_travel_info, tool_weather_forecast] #Define the tools list (in our case, only one tool)
# llm = get_llm_with_responses_api("gpt-5-mini")
llm = get_llm("gpt-5-mini", use_responses_api=True)

# Step 2: REMOVING MANUAL TOOL BINDING
# llm_with_tools = llm_model.bind_tools(TOOLS) => Should be Removed

# ----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------

# Step 3: include a remaining_steps field. This field allows
# the agent to manage how many tool-calling rounds are left in a controlled way:

class AgentState(TypedDict): #Define the agent state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #this is a special type of state that contains the remaining steps of the agent

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------
# a single instantiation of the built-in LangGraph ReAct agent
agent_travel_info = create_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
)

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): 
    #Define the chat loop
    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() 
        if user_input.lower() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=user_input)]} #Create the initial state with a HumanMessage containing the user input
        result = agent_travel_info.invoke(state) #Invoke the graph with the initial state
        response_msg = result["messages"][-1] #Get the last message from the result, which contains the final answer
        print(f"Assistant: {response_msg.content}\n") 

if __name__ == "__main__":
    chat_loop()

#  Running the prebuilt agent
    
# Step 5: Observing and debugging with LangSmith
# A common concern when switching to high-level abstractions is loss of visibility: How
# do we know the agent is actually following the right reasoning steps? While we can
# still debug tool functions directly, the flow inside the agent itself is less exposed. This
# is where LangSmith comes in. LangSmith enables full tracing and inspection of agent
# behavior, including tool calls, LLM reasoning, and intermediate states.
    

# Step 6: Enabling LangSmith tracing



