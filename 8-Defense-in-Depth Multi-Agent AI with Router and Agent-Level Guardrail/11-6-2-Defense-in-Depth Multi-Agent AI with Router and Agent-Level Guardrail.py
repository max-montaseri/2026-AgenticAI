# DEFINING THE GUARDRAIL POLICY
# Our first task is to clearly define what qualifies as in-scope for this assistant. This
# ensures the guardrail has unambiguous decision criteria.

# Implementing more restrictive guardrails at the agent level
# In traditional software development, it’s considered best practice for each class or
# component to validate its own data rather than relying solely on validations at higher
# levels such as the UI. The same principle applies to agent-based systems: each agent
# should enforce its own input guardrails, even if broader checks are already in place at
# the chatbot entry point.

# These agent-level guardrails are often more restrictive than system-wide ones
# because they can account for the specific capabilities and data scope of the individual
# agent. In our case, the following is true:

    #  The travel information agent can only handle queries about Cornwall because
    # its vector store contains data exclusively from that region.

    #  The accommodation booking agent will also be limited to Cornwall for now to
    # keep the assistant’s scope consistent.

# We have two levels of guardrails:
    #  Router-level—This guardrail acts as an early fail-fast filter before any agent logic
    # or tool invocation.

    #  Agent-level—These guardrails provide a “belt-and-suspenders” safeguard to
    # catch out-of-scope requests if the agent is ever called directly or reused in a different
    # context.

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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
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
    
load_dotenv()

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

    print(f"Embedding {len(chunks)} chunks ...")
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
# Guardrail: pre_model_hook to allow only travel-related questions
# -----------------------------------------------------------------------------
class GuardrailDecision(BaseModel): #Define the GuardrailDecision model
    is_travel: bool = Field(
        ...,
        description=(
            "True if the user question is about travel information: destinations, attractions, "
            "lodging (hotels/BnBs), prices, availability, or weather in Cornwall/England."
        ),
    )
    reason: str = Field(..., description="Brief justification for the decision.")

GUARDRAIL_SYSTEM_PROMPT = ( #Define the GUARDRAIL_SYSTEM_PROMPT which constrains the model to only answer travel-related questions
    "You are a strict classifier. Given the user's last message, respond with whether it is "
    "travel-related. Travel-related queries include destinations, attractions, lodging (hotels/BnBs), "
    "room availability, prices, or weather in Cornwall/England."
)

REFUSAL_INSTRUCTION = ( #Define the REFUSAL_INSTRUCTION which is used to politely refuse to answer non-travel-related questions
    "You can only help with travel-related questions (destinations, attractions, lodging, prices, "
    "availability, or weather in Cornwall/England). The user's request is not travel-related. "
    "Politely refuse and briefly explain what topics you can help with."
)


llm_guardrail = llm.with_structured_output(GuardrailDecision) #Use the same base model with structured output for fast, lightweight classification

# Step 1: DEFINING THE CORNWALL-RESTRICTED GUARDRAIL POLICY
# System prompts for classification and refusal behavior

AGENT_GUARDRAIL_SYSTEM_PROMPT = ( 
    """You are a strict classifier. Given the user's last message, 
    respond with whether it is travel-related. Travel-related 
    queries include destinations, attractions, lodging 
    (hotels/BnBs), room availability, prices, or weather in 
    Cornwall/England. Only accept travel-related questions covering 
    Cornwall (England) and reject any questions from other areas in 
    England and from other countries"""
)

AGENT_REFUSAL_INSTRUCTION = ( 
    """You can only help with travel-related questions 
    (destinations, attractions, lodging, prices, 
    availability, or weather in Cornwall/England). The user's 
    request is not travel-related. Or it might be a travel 
    related question but not focusing on Cornwall (England). 
    Politely refuse and briefly explain what 
    topics you can help with."""
)

# Step 2: CREATING THE AGENT-LEVEL GUARDRAIL FUNCTION
# Agent-level guardrail function

# The agent guardrail is implemented as a Python function that takes the current graph
# state and returns either an unchanged state (for valid input) or one modified to
# instruct the LLM to issue a refusal.

# The guardrail_preprocessing_filter_model() function works as a preprocessing filter before the LLM
# sees the user’s query by doing the following:
# 1 It verifies that the latest message is indeed from the user.
# 2 It sends the query, along with a strict classification system prompt, to the guardrail
# LLM.
# 3 If the query is in scope (travel-related and Cornwall-specific), it passes through
# unchanged. Otherwise, the function prepends a refusal instruction so the agent
# politely declines the request.

def guardrail_preprocessing_filter_model(state: dict):
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if not isinstance(last_msg, HumanMessage): #Check if the last message is a HumanMessage (which is the user input)
        return {}

    user_input = last_msg.content
    classifier_messages = [ #Create the classifier messages, including the system prompt and the user input
        SystemMessage(content=AGENT_GUARDRAIL_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]
    decision = llm_guardrail.invoke(classifier_messages)

    if decision.is_travel: #Check if the decision is travel-related. If so, allow normal flow; do not modify inputs
        # Allow normal flow; do not modify inputs
        return {}

    # Inject a refusal instruction ahead of the original messages so the model politely declines
    return {"llm_input_messages": 
        [SystemMessage(content=AGENT_REFUSAL_INSTRUCTION),
        *messages]} #If the decision is not travel-related, inject a refusal instruction ahead of the original messages so the model politely declines

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
    messages = state["messages"] 
    last_msg = messages[-1] if messages else None 
    if isinstance(last_msg, HumanMessage):
        user_input = last_msg.content 

        # Guardrail classification at routing time
        classifier_messages = [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT), #Define the guardrail decision prompt
            HumanMessage(content=user_input),
        ]
        decision = llm_guardrail.invoke(classifier_messages) #Invoke the guardrail model, which returns a GuardrailDecision object
        if not decision.is_travel: #Check if the decision is not travel-related
            # Return refusal directly as an AI message and shortcut to END via a dedicated node
            refusal_text = ( #Define the refusal text
                "Sorry, I can only help with travel-related questions (destinations, attractions, "
                "lodging, prices, availability, or weather in Cornwall/England). "
                "Please rephrase your request to be travel-related."
            )
            return Command( #Return the command to set a refusal message in the state and go to the guardrail refusal node
                update={"messages": [AIMessage(content=refusal_text)]},
                goto="guardrail_refusal",
            ) 

        router_messages = [ 
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]
        router_response = llm_router.invoke(router_messages) 
        agent_name = router_response.agent.value 
        return Command(update=state, goto=agent_name) 
    
    return Command(update=state, goto=AgentType.agent_travel_info) 

# -----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Build the travel info assistant React Agent
# ----------------------------------------------------------------------------

# Step 3: INJECTING THE GUARDRAIL INTO THE AGENTS

# LangGraph’s ReAct agents support pre-model hooks (pre_model_hook) and postmodel
# hooks (post_model_hook), allowing us to intercept and manipulate inputs or
# outputs. While these hooks can be used for tasks such as summarizing long inputs or
# sanitizing outputs, here we’ll focus solely on input-side guardrails. To enable the
# Cornwall restriction, we simply pass guardrail_preprocessing_filter_model to both the travel information
# agent and the accommodation booking agent, as shown in the following listings.

# Step 3-1: Travel information agent with Cornwall guardrail
agent_travel_info = create_agent(
    model=llm,
    tools=TOOLS,
    state_schema=AgentState,
    prompt="""You are a helpful assistant that can search travel 
    information and get the weather forecast. Only use the tools 
    to find the information you need (including town names).""",
    pre_model_hook=guardrail_preprocessing_filter_model, #Guardrail to check if the user input is travel-related and focusing on Cornwall (England)
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

# Step 3-2: Accommodation booking agent with Cornwall guardrail
agent_accommodation_booking = create_agent( #Guardrail to check if the user input is travel-related and focusing on Cornwall (England)
    model=llm,
    tools=BOOKING_TOOLS,
    state_schema=AgentState,
    prompt="""You are a helpful assistant that can check hotel 
    and BnB room availability and price for a destination in 
    Cornwall. You can use the tools to get the information you 
    need. If the users does not specify the accommodation type,
    you should check both hotels and BnBs.""",
    pre_model_hook=guardrail_preprocessing_filter_model,
)

# -----------------------------------------------------------------------------
# Build the LangGraph graph with router, agent_travel_info, and agent_accommodation_booking
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Guardrail refusal node (no-op, used to shortcut to END)
# -----------------------------------------------------------------------------
def guardrail_refusal_node(state: AgentState): 
    #Define the guardrail refusal node, which is a no-op node that is used to shortcut to END 
    return {}

graph = StateGraph(AgentState) 
graph.add_node("router_agent", router_agent_node) 
graph.add_node("agent_travel_info", agent_travel_info) 
graph.add_node("agent_accommodation_booking", agent_accommodation_booking) 
graph.add_node("guardrail_refusal", guardrail_refusal_node) #adding the guardrail refusal node

graph.add_edge("agent_travel_info", END) 
graph.add_edge("agent_accommodation_booking", END) 
graph.add_edge("guardrail_refusal", END) #Adding the edge from the guardrail refusal node to the end

graph.set_entry_point("router_agent") 

checkpointer = InMemorySaver() 
multi_agent_supervisor_travel_assistant = graph.compile(checkpointer=checkpointer) 

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

def chat_loop(): #Define the chat loop
    thread_id=uuid.uuid1() #Create a unique thread id
    print(f'Thread ID: {thread_id}') 
    config={"configurable": {"thread_id": thread_id}}

    print("UK Travel Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip() 
        if user_input.lower() in {"exit", "quit"}: 
            break
        state = {"messages": [HumanMessage(content=user_input)]}
        result = multi_agent_supervisor_travel_assistant.invoke(state, config=config) #Invoke the graph with the state and the config
        response_msg = result["messages"][-1] #Get the last message from the result, which contains the final answer
        print(f"Assistant: {response_msg.content}\n") #Print the assistant's final answer, from the content of the last message

if __name__ == "__main__":
    chat_loop() 


# TESTING THE CORNWALL GUARDRAIL
# Run 
# This project in debug mode, placing a breakpoint on the llm_guardrail invocation
# inside guardrail_preprocessing_filter_model(). Then, try the following:
    
# UK Travel Assistant (type 'exit' to quit)
# You: Can you give me some travel tips for Liverpool (UK)?
    
# When paused at the breakpoint, inspect decision.is_travel—it should be False
# because the query isn’t Cornwall-specific. Execution will then prepend the refusal
# instruction, resulting in output like this:
    
# Assistant: [{'type': 'text', 'text': 'Sorry—I can only help with travel
# questions focused on Cornwall (England), such as destinations, attractions,
# lodging, prices/availability, and local weather. If you’d like tips for places
# like St Ives, Newquay, Falmouth, Penzance, Padstow, or Truro, tell me your
# interests and dates/budget and I’ll tailor suggestions.', 'annotations': []}]
    
# With this, we now have two layers of defense:
    #  A router-level guardrail that quickly rejects any nontravel queries
    
    #  Agent-level guardrails that enforce Cornwall-specific scope for travel and
    # accommodation requests
    
# Our agentic workflow is now protected from both irrelevant and out-of-coverage queries,
# making the system safer, more reliable, and potentially more cost-efficient by preventing
# misuse through questions the chatbot isn’t designed to handle.