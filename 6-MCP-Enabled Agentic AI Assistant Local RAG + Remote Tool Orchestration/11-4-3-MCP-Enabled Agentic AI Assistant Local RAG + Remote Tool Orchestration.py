# Integrating the Weather MCP tool into an agent
# Integrating the live weather tool from the remote MCP server into the travel information agent, allowing it to consume
# real AccuWeather data alongside its existing local capabilities

# Preparing the travel agent for live weather data with a client that connects to the AccuWeather 
# MCP server and retrieves the remote tool dynamically.

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import asyncio
import operator
import os
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv, find_dotenv


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
# The function from langchain.agents import create_agent is the modern, official replacement for 
# from langgraph.prebuilt import create_react_agent.
from langchain.agents import create_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph_supervisor.supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
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
def get_llm_with_responses_api(llm_model):
    llm_model = get_llm_model(llm_model)
    llm = ChatOpenAI(model=llm_model, use_responses_api=True)
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
load_project_dotenv()

# -----------------------------------------------------------------------------
# Step 1: Prepare Travel Info knowledge base (travel information vector store) at startup
# -----------------------------------------------------------------------------

UK_DESTINATIONS = [
    "Cornwall",
    "North_Cornwall",
    "South_Cornwall",
    "West_Cornwall",
]

async def rag_build_vectorstore(destinations: Sequence[str]) -> Chroma:
    # Function to build the vectorstore and return a reference to the vectorstore client
    """Download WikiVoyage pages and create a Chroma vector store."""
    urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations] #Load the destination pages asynchronously from the web into a list of documents
    # Disable SSL verification (development only):
    # If you're just experimenting, you can disable SSL verification. Do not do this in production.
    loader = AsyncHtmlLoader(urls, verify_ssl=False)
    print("Downloading destination pages ...")
    docs = await loader.aload()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = sum([splitter.split_documents([d]) for d in docs], [])

    print(f"Embedding {len(chunks)} chunks ...") #E
    vectordb_client = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())
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
# Step 2: Define Travel Info tool
# ----------------------------------------------------------------------------

@tool(description="Search travel information about destinations in England.") #Define the tool using the @tool decorator
def tool_search_travel_info(query: str) -> str: 
    #Define the tool function, which takes a query, performs a semantic search and returns a string response from the vectorstore
    """Search embedded WikiVoyage content for information about destinations in England."""
    docs = travel_info_rag_retriever.invoke(query) #Perform a semantic search on the vectorstore and return the top 4 results
    top = docs[:4] if isinstance(docs, list) else docs
    return "\n---\n".join(d.page_content for d in top) #Joins the top 4 results into a single string

# ----------------------------------------------------------------------------
# Step 3: Configure LLM with tool awareness
# ----------------------------------------------------------------------------

# Step 3-1: Integrating the AccuWeather MCP tool
# Start by implementing an asynchronous function to instantiate a client for the Accu-
# Weather MCP server. This will return the tools exposed by the server (in this case, just one):
async def mcp_tool_get_accuweather(): 
    #Define the function to get the AccuWeather tools as an async function
    mcp_client = MultiServerMCPClient({ #Instantiate the MultiServerMCPClient
        "accuweather": { #Register the AccuWeather MCP server
            "url": "http://127.0.0.1:8020/accu-mcp-server",
            "transport": "streamable_http"
        }
    })
    return await mcp_client.get_tools() #Return the AccuWeather tools exposed by the MCP server

# Step 3-2: Updating the agent chat loop
# Because the agent now calls out to remote tools, we need to adapt the main chat
# loop to support asynchronous tool invocation:

async def chat_loop(agent): 
    #Define the chat loop as an async function
    print("UK Travel Assistant (type 'exit' to quit)")
    while True: #Start the chat loop
        user_input = input("You: ").strip() 
        if user_input.lower() in {"exit", "quit"}: 
            break
        state = {"messages": [HumanMessage(
            content=user_input)]} #Create the initial state with a HumanMessage containing the user input
        result = await agent.ainvoke(state) #Invoke the agent with the initial state, asyncronously
        response_msg = result["messages"][-1] #Get the last message from the result, which contains the final answer
        print(
           f"Assistant: {response_msg.content}\n")

# Step 3-3: Combining local and remote tools
# In ourasync main() function,we now retrieve the AccuWeather tools and combine
# them with our local semantic search tool, as shown in the following listing.

class AgentState(TypedDict): #Define the AgentState class
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps


# NOTE The main function and chat loop are both now asynchronous, allowing
# our agent to use local and MCP tools with minimal effort.

async def main():
    accuweather_tools = \
        await mcp_tool_get_accuweather() #Get the AccuWeather MCP server tools
    tools = [tool_search_travel_info, *accuweather_tools] #Combine the local tool_search_travel_info tool with the AccuWeather MCP server tools
    llm = get_llm(model="gpt-5-mini", use_responses_api=True)

    agent_travel_info = create_agent( #Create the agent_travel_info
        model=llm,
        tools=tools,
        state_schema=AgentState,
        name="agent_travel_info",
        prompt="""You are a helpful assistant that can 
        search travel information and get the weather forecast. 
        Only use the tools to find the information you need 
        (including town names).""",
    )
    await chat_loop(agent_travel_info) #Start the chat loop

if __name__ == "__main__":
    asyncio.run(main()) #Run the main function, asyncronously 


# Step 3-4: Testing and verification
# With the updated code in place, we can now run the this .py script in debug
# mode. Place a breakpoint at the line where the language model (llm_model) is instantiated,
# and inspect the tools list. We should see both the local and remote tools in the
# output:
    
# As We can see, the AccuWeather MCP server is now invoked by our agent. We can
# also check the terminal in which the MCP server is running and inspect the Lang-
# Smith trace to confirm the tool was called as expected.


# Step 3-5: Using the agent for complex queries
# Finally, experiment with more advanced reasoning-based queries that combine travel
# information with live weather data. Be sure to adapt the questions to the current season.

# For example:
# You: Suggest two beach Cornwall towns with nice weather
# Assistant: [{'type': 'text', 'text': 'Two beach towns in Cornwall are Newquay
# and St Ives. However, currently, Newquay is experiencing light rain with a
# temperature of 17°C, and St Ives has hazy sunshine with a temperature of 6°C.
# If you prefer nicer weather, St Ives would be the better choice at the
# moment.', 'annotations': []}]

# We can continue experimenting with queries like this:
# You: Suggest two beach Cornwall towns with nice weather; keep trying until
# you find two with nice weather

# By integrating the weather tool exposed by our MCP server, we’ve made our agent
# capable of delivering genuinely real-time, actionable information. This not only
# demonstrates the power of MCP but also how external tools can be combined with
# local agent skills for richer, more useful applications.