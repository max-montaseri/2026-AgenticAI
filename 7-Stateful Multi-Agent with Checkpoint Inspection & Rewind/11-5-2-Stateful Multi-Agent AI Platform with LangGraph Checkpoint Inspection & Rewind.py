# Adding short-term memory to our travel assistant
# We’ll demonstrate how LangGraph’s persistence and checkpointing work
# We’ll add persistence features to this copy so we can compare it with the original.

# Rewinding the state to a past checkpoint
# To better understand how LangGraph manages conversational memory, we’ll simulate
# what happens internally when restoring from a checkpoint as follows:
    # 1 Ask the chatbot a question.
    # 2 Retrieve the last checkpoint from the checkpointer.
    # 3 Rehydrate the graph state to that checkpoint.
    # 4 Ask a follow-up question that depends on that restored context.
# This is effectively what LangGraph does automatically when we pass the same
# thread_id on subsequent turns.

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

import os
import uuid
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict
from dotenv import load_dotenv, find_dotenv
import random
from enum import Enum
from pydantic import BaseModel, Field


from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.tools import tool
# The function from langchain.agents import create_agent is the modern, official replacement for 
# from langgraph.prebuilt import create_react_agent.
from langchain.agents import create_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


#region Settings
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
def load_project_dotenv():
    # load the environment variables from the .env 
    load_dotenv()
#endregion

load_project_dotenv()

# -----------------------------------------------------------------------------
# 1. Prepare Travel Info knowledge base (travel information vector store) at startup
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
# 2-1 Define Travel Info tool
# ----------------------------------------------------------------------------

@tool(description="Search travel information about destinations in England.") #Define the tool using the @tool decorator
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

# llm_model = ChatOpenAI(model="gpt-5", #Instantiate the LLM model with the gpt-5 model
#                        use_responses_api=True, #Use the Responses API                      
#                        use_previous_response_id=True) #Use the previous response ID to continue the conversation
llm = get_llm("gpt-5", use_responses_api=True, use_previous_response_id=True)

# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict): #Define the agent state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    remaining_steps: RemainingSteps #this is a special type of state that contains the remaining steps of the agent

# -----------------------------------------------------------------------------
# AgentType Enum and Structured Output Model
# -----------------------------------------------------------------------------
class AgentType(str, Enum):
    agent_travel_info = "agent_travel_info"
    agent_accommodation_booking = "agent_accommodation_booking"

class AgentTypeOutput(BaseModel): 
    agent: AgentType = Field(..., description="Which agent should handle the query?")

# Structured LLM for routing
llm_router = llm.with_structured_output(AgentTypeOutput)

# -----------------------------------------------------------------------------
# Router Agent System Prompt Constant
# -----------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = (
    "You are a router. Given the following user message, decide if it is a travel information question (about destinations, attractions, or general travel info) "
    "or an accommodation booking question (about hotels, BnBs, room availability, or prices).\n"
    "If it is a travel information question, respond with 'agent_travel_info'.\n"
    "If it is an accommodation booking question, respond with 'agent_accommodation_booking'."
)

# -----------------------------------------------------------------------------
# Router Agent Node for LangGraph (with structured output)
# -----------------------------------------------------------------------------
def router_agent_node(state: AgentState) -> Command[AgentType]:
    """Router node: decides which agent should handle the user query."""
    messages = state["messages"] #Get the messages from the state
    last_msg = messages[-1] if messages else None #Get the last message from the messages list
    if isinstance(last_msg, HumanMessage): #Check if the last message is a HumanMessage
        user_input = last_msg.content #Get the content of the last message
        router_messages = [ #Create the router messages, including the system prompt and the user input
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        router_response = llm_router.invoke(router_messages) #Invoke the router model, which returns the relevant agent name
        agent_name = router_response.agent.value #Get the agent name from the router response
        return Command(update=state, goto=agent_name) #Return the command to update the state and go to the agent
    
    return Command(update=state, goto=AgentType.agent_travel_info) #If the last message is not a HumanMessage, return the command to update the state and go to the agent_travel_info (default agent)

# -----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

agent_travel_info = create_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can search travel information and get the weather forecast. Only use the tools to find the information you need (including town names).",
)

# -----------------------------------------------------------------------------
# 2-3 SQLDatabaseToolkit for Hotel Booking (SQLite)
# -----------------------------------------------------------------------------
hotel_db = SQLDatabase.from_uri("sqlite:///hotel_db/cornwall_hotels.db")
hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=llm)
hotel_db_toolkit_tools = hotel_db_toolkit.get_tools()

# -----------------------------------------------------------------------------
# 2-4 ToolBnBBookingService (Mock REST API client)
# -----------------------------------------------------------------------------

class DictToolBnBOffer(TypedDict): #Define the return type of the BnB availability tool
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float

class ToolBnBBookingService: #Define the BnB availability tool
    @staticmethod
    def tool_get_offers_near_town(town: str, num_rooms: int) -> List[DictToolBnBOffer]: #Call the BnB booking service to get the offers
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = [ #Mocked BnB offers
            # Newquay
            {"bnb_id": 1, "bnb_name": "Seaside BnB", "town": "Newquay", "available_rooms": 3, "price_per_room": 80.0},
            {"bnb_id": 2, "bnb_name": "Surfside Guesthouse", "town": "Newquay", "available_rooms": 2, "price_per_room": 85.0},
            # Falmouth
            {"bnb_id": 3, "bnb_name": "Harbour View BnB", "town": "Falmouth", "available_rooms": 4, "price_per_room": 78.0},
            {"bnb_id": 4, "bnb_name": "Seafarer's Rest", "town": "Falmouth", "available_rooms": 1, "price_per_room": 90.0},
            # St Austell
            {"bnb_id": 5, "bnb_name": "Garden Gate BnB", "town": "St Austell", "available_rooms": 2, "price_per_room": 82.0},
            {"bnb_id": 6, "bnb_name": "Coastal Cottage BnB", "town": "St Austell", "available_rooms": 3, "price_per_room": 88.0},
            # Penzance
            {"bnb_id": 7, "bnb_name": "Penzance Pier BnB", "town": "Penzance", "available_rooms": 2, "price_per_room": 95.0},
            {"bnb_id": 8, "bnb_name": "Cornish Charm BnB", "town": "Penzance", "available_rooms": 3, "price_per_room": 87.0},
            # Camborne
            {"bnb_id": 9, "bnb_name": "Camborne Corner BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 75.0},
            {"bnb_id": 10, "bnb_name": "Rose Cottage BnB", "town": "Camborne", "available_rooms": 2, "price_per_room": 79.0},
            # Hayle
            {"bnb_id": 11, "bnb_name": "Hayle Haven BnB", "town": "Hayle", "available_rooms": 3, "price_per_room": 83.0},
            {"bnb_id": 12, "bnb_name": "Dune View BnB", "town": "Hayle", "available_rooms": 1, "price_per_room": 81.0},
            # Land's End
            {"bnb_id": 13, "bnb_name": "Land's End Lookout BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 100.0},
            {"bnb_id": 14, "bnb_name": "Atlantic Edge BnB", "town": "Land's End", "available_rooms": 2, "price_per_room": 105.0},
            # Bude
            {"bnb_id": 15, "bnb_name": "Bude Beach BnB", "town": "Bude", "available_rooms": 2, "price_per_room": 77.0},
            {"bnb_id": 16, "bnb_name": "Cliffside BnB", "town": "Bude", "available_rooms": 3, "price_per_room": 80.0},
            # Padstow
            {"bnb_id": 17, "bnb_name": "Padstow Harbour BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 92.0},
            {"bnb_id": 18, "bnb_name": "Fisherman's Rest BnB", "town": "Padstow", "available_rooms": 2, "price_per_room": 89.0},
            # St Ives
            {"bnb_id": 19, "bnb_name": "St Ives Bay BnB", "town": "St Ives", "available_rooms": 3, "price_per_room": 97.0},
            {"bnb_id": 20, "bnb_name": "Artists' Retreat BnB", "town": "St Ives", "available_rooms": 2, "price_per_room": 102.0},
            # Looe
            {"bnb_id": 21, "bnb_name": "Looe Riverside BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 84.0},
            {"bnb_id": 22, "bnb_name": "Harbour Lights BnB", "town": "Looe", "available_rooms": 2, "price_per_room": 86.0},
            # Polperro
            {"bnb_id": 23, "bnb_name": "Polperro Cove BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 91.0},
            {"bnb_id": 24, "bnb_name": "Smuggler's Rest BnB", "town": "Polperro", "available_rooms": 2, "price_per_room": 93.0},
            # Mevagissey
            {"bnb_id": 25, "bnb_name": "Mevagissey Harbour BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 90.0},
            {"bnb_id": 26, "bnb_name": "Seafarer's BnB", "town": "Mevagissey", "available_rooms": 2, "price_per_room": 88.0},
            # Port Isaac
            {"bnb_id": 27, "bnb_name": "Port Isaac View BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 99.0},
            {"bnb_id": 28, "bnb_name": "Fisherman's Cottage BnB", "town": "Port Isaac", "available_rooms": 2, "price_per_room": 101.0},
            # Fowey
            {"bnb_id": 29, "bnb_name": "Fowey Quay BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 94.0},
            {"bnb_id": 30, "bnb_name": "Riverside Rest BnB", "town": "Fowey", "available_rooms": 2, "price_per_room": 96.0},
        ]
        offers = [offer for offer in mock_bnb_offers if offer["town"].lower() == town.lower() and offer["available_rooms"] >= num_rooms]
        return offers
    
# -----------------------------------------------------------------------------
# BnB Availability Tool
# -----------------------------------------------------------------------------
@tool(description="Check BnB room availability and price for a destination in Cornwall.") #Define the BnB availability tool
def tool_check_bnb_availability(destination: str, num_rooms: int) -> List[Dict]: 
    #Define the input and return type of the BnB availability tool
    """Check BnB room availability and price for the requested destination and number of rooms."""
    offers = ToolBnBBookingService.tool_get_offers_near_town(destination, num_rooms)
    if not offers:
        return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
    return offers

# -----------------------------------------------------------------------------
# Accommodation Booking Agent
# -----------------------------------------------------------------------------
BOOKING_TOOLS = hotel_db_toolkit_tools + [tool_check_bnb_availability] 
#Define the booking tools, which are the tools from the hotel database toolkit and the BnB availability tool

agent_accommodation_booking = create_agent( #Create the accommodation booking agent
    model=llm,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. You can use the tools to get the information you need. If the users does not specify the accommodation type, you should check both hotels and BnBs.",
)

# -----------------------------------------------------------------------------
# Build the LangGraph graph with router, agent_travel_info, and agent_accommodation_booking
# -----------------------------------------------------------------------------
graph = StateGraph(AgentState) #Define the graph
graph.add_node("router_agent", router_agent_node) #Adding the router agent node
graph.add_node("agent_travel_info", agent_travel_info) #Adding the travel info agent node
graph.add_node("agent_accommodation_booking", agent_accommodation_booking) #Adding the accommodation booking agent node

graph.add_edge("agent_travel_info", END) #Adding the edge from the travel info agent to the end
graph.add_edge("agent_accommodation_booking", END) #Adding the edge from the accommodation booking agent to the end

graph.set_entry_point("router_agent") #Set the entry point to the router agent

checkpointer = InMemorySaver() #Instantiate the in-memory checkpointer
multi_agent_supervisor_travel_assistant = graph.compile(checkpointer=checkpointer) #Compile the graph with the in-memory checkpointer

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

# STEP 1: UPDATING THE CHAT LOOP FOR STATE INSPECTION
def chat_loop(): 
    thread_id=uuid.uuid1() #Create a unique thread id
    print(f'Thread ID: {thread_id}') 
    config={"configurable": 
       {"thread_id": thread_id}} #Create a config with the thread id

    user_input = input("You: ").strip() #Create the initial state with a HumanMessage containing the user input

    question = {"messages": 
        [HumanMessage(content=user_input)]} #Set the state with the HumanMessage
    result = multi_agent_supervisor_travel_assistant.invoke(
        question, config=config) #Invoke the graph with the state and the config
    response_msg = result["messages"][-1] #Get the last message from the result, which contains the final answer
    print(
       f"Assistant: {response_msg.content}\n") #Print the assistant's final answer, from the content of the last message

    state_history = multi_agent_supervisor_travel_assistant.get_state_history(
        config) #Get the state history from the graph
    state_history_list = list(state_history) #I
    print(f'State history: {state_history_list}') #Print the state history

    # STEP 2: REHYDRATING FROM A SPECIFIC CHECKPOINT
    # Starting from the most recent snapshot
    last_snapshot = list(state_history_list)[0] #Get the last snapshot from the state history
    print(f'Last snapshot: {last_snapshot.config}')

    # Extract the thread_id and checkpoint_id:
    thread_id = last_snapshot.config[
        "configurable"]["thread_id"] #Get the thread id from the last snapshot
    last_checkpoint_id = last_snapshot.config[
        "configurable"]["checkpoint_id"] #Get the checkpoint id from the last snapshot

    # Build a new config pointing to that checkpoint:
    new_config = {"configurable": #Create a new config with the thread id and the checkpoint id
              {"thread_id": thread_id, 
               "checkpoint_id": last_checkpoint_id}}
    
    # Retrieve the state at this checkpoint to confirm it matches expectations:
    retrieved_snapshot = multi_agent_supervisor_travel_assistant.get_state(
        new_config) #Get the snapshot from the graph with the new config referencing the last checkpoint
    print(
       f'Retrieved snapshot: {retrieved_snapshot}') #Print the retrieved snapshot

    # STEP 3: RESUMING FROM THE RESTORED STATE
    # To rewind the graph to that point, use the following:
    multi_agent_supervisor_travel_assistant.invoke(None, 
    config=new_config) #Rewind the graph to the last checkpoint

    new_question = {"messages": [HumanMessage(
        content="What is the weather in the same town?")]}
    result = multi_agent_supervisor_travel_assistant.invoke(new_question, 
        config=new_config) #Invoke the graph with the new question referencing content from the last checkpoint
    response_msg = result["messages"][-1] #Get the last message from the result, which contains the final answer

    print(
       f"Assistant: {response_msg.content}\n") #Print the assistant's final answer, from the content of the last message

if __name__ == "__main__":
    chat_loop() 

# STEP 4: RUNNING THE EXAMPLE AND VIEWING STATE HISTORY
# Run in debug mode, and enter the usual question (remember, the weather is still randomized)