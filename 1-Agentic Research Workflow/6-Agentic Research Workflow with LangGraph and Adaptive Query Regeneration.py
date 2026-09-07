import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from langchain_text_splitters import TokenTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
import json

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from duckduckgo_search.exceptions import DuckDuckGoSearchException # => pip install -U duckduckgo-search

import requests
from bs4 import BeautifulSoup

import re

from prompts_for_agentic_workflows_with_LangGraph import (ASSISTANT_SELECTION_PROMPT_TEMPLATE, 
    RESEARCH_REPORT_PROMPT_TEMPLATE, WEB_SEARCH_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE)

from dataclasses import dataclass, field
from typing import List, Dict, Any, TypedDict, Optional

import time
import random

from langgraph.graph import StateGraph, END

#region DEFINE THE STATE (Model)

# from dataclasses import dataclass, field
# from typing import List, Dict, Any, TypedDict, Optional

# Step 1- DEFINE THE STATE (Model):
# The first step is to design the state structure that will flow through the graph. A welldefined
# state helps us keep track of data across all nodes.

# We model a composite state using inner types. This state structure clearly
# defines the data available at each stage, reducing ambiguity and simplifying debugging.

@dataclass
# Step 1-1: Define typed dictionaries for state handling
class AssistantInfo(TypedDict):
    assistant_type: str
    assistant_instructions: str
    user_question: str

class SearchQuery(TypedDict):
    search_query: str
    user_question: str

class SearchResult(TypedDict):
    result_url: str
    search_query: str
    user_question: str
    is_fallback: Optional[bool]

class SearchSummary(TypedDict):
    summary: str
    result_url: str
    user_question: str
    is_fallback: Optional[bool]

class ResearchReport(TypedDict):
    report: str

# Step 1-2: Graph state
class ResearchState(TypedDict):
    user_question: str
    assistant_info: Optional[AssistantInfo]
    search_queries: Optional[List[SearchQuery]]
    search_results: Optional[List[SearchResult]]
    search_summaries: Optional[List[SearchSummary]]
    research_summary: Optional[str]
    final_report: Optional[str]
    used_fallback_search: Optional[bool]
    relevance_evaluation: Optional[Dict[str, Any]]
    should_regenerate_queries: Optional[bool]
    iteration_count: Optional[int]
#endregion

#region GRAPH STRUCTURE and Run the GRAPH
def create_research_graph() -> StateGraph:
    # from langgraph.graph import StateGraph, END

    # DEFINE THE GRAPH STRUCTURE
    # With node functions in place, we create the graph and define how the nodes connect, establishing the execution order and data flow
    # Unlike a simple linear chain, this version of the graph introduces a new node for relevance
    # evaluation and a conditional edge that dynamically alters the flow based on the relevance of the search results.    
    """
    Create the LangGraph research graph that coordinates the agents.
    """
    # Step 3-1: Define the graph
    graph = StateGraph(ResearchState)
    
    # Step 3-2: Add nodes to the graph
    graph.add_node("select_assistant", SubAgentAssistantSelector)
    graph.add_node("generate_search_queries", SubAgentGenerateSearchQueries)
    graph.add_node("perform_web_searches", SubAgentPerformWebSearches)
    graph.add_node("summarize_search_results", SubAgentSummarizeSearchResults)
    graph.add_node("evaluate_search_relevance", SubAgentEvaluateSearchRelevance)
    graph.add_node("write_research_report", SubAgentResearchReportWrite)
    
    # Step 3-3: Define the conditional routing function for relevance evaluation
    def route_based_on_relevance(state: Dict[str, Any]) -> str:
        """
        Route to either generate new search queries or continue to report writing
        based on the relevance evaluation.
        """
        # 3-3-1 Get the current iteration count
        iteration_count = state.get("iteration_count", 0)
        
        # 3-3-2 Increment the iteration count
        new_iteration_count = iteration_count + 1
        
        # 3-3-3 Update the state with the new iteration count
        state["iteration_count"] = new_iteration_count
        
        # 3-3-4 Check if we've reached the maximum number of iterations (3)
        if new_iteration_count >= 3:
            print(f"Reached maximum iterations ({new_iteration_count}). Proceeding to write report with current results.")
            return "write_research_report"
        
        # Otherwise, check if we should regenerate queries
        if state.get("should_regenerate_queries", False):
            print(f"Iteration {new_iteration_count}: Regenerating search queries.")
            return "generate_search_queries"
        else:
            print(f"Iteration {new_iteration_count}: Search results are relevant. Proceeding to write report.")
            return "write_research_report"
    
    # Step 3-4: Define the flow of the graph (the edges between nodes)
    graph.add_edge("select_assistant", "generate_search_queries")
    graph.add_edge("generate_search_queries", "perform_web_searches")
    graph.add_edge("perform_web_searches", "summarize_search_results")
    graph.add_edge("summarize_search_results", "evaluate_search_relevance")
    
    # Step 3-5: Add conditional routing based on relevance evaluation (conditional edge)
    graph.add_conditional_edges(
        "evaluate_search_relevance",
        route_based_on_relevance,
        {
            "generate_search_queries": "generate_search_queries",
            "write_research_report": "write_research_report"
        }
    )
    
    # Step 3-6: Add the END edge
    graph.add_edge("write_research_report", END)
    
    # Step 3-7: Set the entry point
    graph.set_entry_point("select_assistant")
    
    return graph   

def run_research(question: str) -> str:
    # STEP 4: COMPILE AND RUN THE GRAPH    
    """
    Run the research graph with a user question.
    
    Args:
        question: The user's research question
        
    Returns:
        The final research report
    """
    # Step 4-1: Create the graph
    research_graph = create_research_graph()
    
    # Step 4-2: Compile the graph
    app = research_graph.compile()
    
    # Step 4-3: Initialize the state
    initial_state = {
        "user_question": question,
        "assistant_info": None,
        "search_queries": None,
        "search_results": None,
        "search_summaries": None,
        "research_summary": None,
        "final_report": None,
        "used_fallback_search": False,
        "relevance_evaluation": None,
        "should_regenerate_queries": None,
        "iteration_count": 0
    }
    
    # Step 4-4: Run the graph
    result = app.invoke(initial_state)
    
    # Step 4-5: Extract and return the final report
    return result["final_report"]

#endregion

#region Sub-agents

def SubAgentAssistantSelector(state: Dict[str, Any]) -> Dict[str, Any]:
    # from langchain_core.output_parsers import StrOutputParser
    # import json
    # from typing import Dict, Any

    # CONVERT COMPONENTS TO Agent NODE FUNCTIONS
    # We convert each component into a node function. Each function takes the current state, 
    # processes it, and returns updated state information
        
    # Assistant Selector Agent Node Function (Component): Determines which type of 
    # research assistant to use based on the user’s question

    """
    Select the appropriate research assistant based on the user question.
    """
    user_question = state["user_question"]
    
    # 1- Use the LLM to select an assistant: Format the prompt with the user question
    prompt = ASSISTANT_SELECTION_PROMPT_TEMPLATE.format(user_question=user_question)
    
    # 2- Get the LLM response
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content
    
    # 3- Parse the response to get the assistant info
    try:
        # 3-1 Extract the JSON part from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        # 3-2 Parse the JSON
        assistant_info = json.loads(json_str)
        
        # 3-2 Return the updated state
        return {"assistant_info": assistant_info}
    except Exception as e:
        # Fallback to a default assistant if parsing fails
        default_assistant = {
            "assistant_type": "General research assistant",
            "assistant_instructions": "You are a general research AI assistant. Your main purpose is to draft comprehensive, informative, unbiased, and well-structured reports on given topics.",
            "user_question": user_question
        }
        return {"assistant_info": default_assistant}

def SubAgentResearchReportWrite(state: Dict[str, Any]) -> Dict[str, Any]:
     # from typing import Dict, Any
    
    # CONVERT COMPONENTS TO Agent NODE FUNCTIONS
    # We convert each component into a node function. Each function takes the current state, 
    # processes it, and returns updated state information

    # Report Writer Agent Node Function (Component): Compiles the final research report 
    # using the relevant summaries    
    """
    Write a research report based on the summarized search results.
    """
    research_summary = state["research_summary"]
    user_question = state["user_question"]
    
    # Format the prompt
    prompt = RESEARCH_REPORT_PROMPT_TEMPLATE.format(
        research_summary=research_summary,
        user_question=user_question
    )
    
    # Get the LLM response
    llm = get_llm()
    response = llm.invoke(prompt)
    report = response.content
    
    # Return the updated state
    return {"final_report": report}    




    # Query Generator Agent Node Function (Component): Creates search queries derived from the user’s input

def SubAgentGenerateSearchQueries(state: Dict[str, Any]) -> Dict[str, Any]:
    # Query Generator Agent Node Function (Component): Creates search queries derived 
    # from the user’s input
    NUM_SEARCH_QUERIES = 3

    """
    Generate search queries based on the assistant instructions and user question.
    Uses different strategies based on iteration count to ensure variety.
    """
    assistant_info = state["assistant_info"]
    user_question = state["user_question"]
    assistant_instructions = assistant_info["assistant_instructions"]
    
    # Get the current iteration count
    iteration_count = state.get("iteration_count", 0)
    
    # Check if this is a regeneration (we already have search queries and relevance evaluation)
    previous_queries = state.get("search_queries", [])
    relevance_evaluation = state.get("relevance_evaluation", None)
    
    # 1- Uses the LLM to create queries
    #   Format the prompt based on iteration count
    if iteration_count == 0:
        # First-time query generation
        print("Generating initial search queries...")
        prompt = WEB_SEARCH_PROMPT_TEMPLATE.format(
            assistant_instructions=assistant_instructions,
            user_question=user_question,
            num_search_queries=NUM_SEARCH_QUERIES
        )
    elif iteration_count == 1:
        # Second iteration - more specific queries
        print("First regeneration: Creating more specific queries...")
        previous_query_list = ", ".join([q["search_query"] for q in previous_queries])
        relevance_percentage = relevance_evaluation.get("relevance_percentage", 0) if relevance_evaluation else 0
        relevance_explanation = relevance_evaluation.get("explanation", "No explanation provided") if relevance_evaluation else ""
        
        prompt = f"""
        {assistant_instructions}

        You are generating new search queries because the previous queries did not yield sufficiently relevant results.
        
        Original question: {user_question}
        
        Previous search queries: {previous_query_list}
        
        Relevance evaluation: {relevance_percentage}% relevant
        Explanation: {relevance_explanation}
        
        Please generate {NUM_SEARCH_QUERIES} NEW and DIFFERENT web search queries that are MORE SPECIFIC and TARGETED 
        to gather relevant information on the original question. 
        
        IMPORTANT: DO NOT repeat or rephrase the previous queries. Create completely different approaches to finding information.
        
        You must respond with a list of queries in the following format:
        [
            {{"search_query": "query1", "user_question": "{user_question}" }},
            {{"search_query": "query2", "user_question": "{user_question}" }},
            {{"search_query": "query3", "user_question": "{user_question}" }}
        ]
        """
    else:
        # Third or later iteration - completely different approach
        print(f"Iteration {iteration_count}: Using alternative search strategies...")
        all_previous_queries = ", ".join([q["search_query"] for q in previous_queries])
        
        prompt = f"""
        {assistant_instructions}

        You are generating search queries for the FINAL attempt to find relevant information.
        
        Original question: {user_question}
        
        All previous search queries that DID NOT yield relevant results: {all_previous_queries}
        
        For this final attempt, take a completely different angle. Consider:
        1. Breaking down the question into smaller, more focused sub-questions
        2. Using technical or specialized terms related to the topic
        3. Searching for expert opinions or academic perspectives
        4. Looking for case studies or specific examples
        5. Exploring historical context or background information
        
        CRITICAL INSTRUCTIONS:
        1. DO NOT repeat or rephrase ANY previous queries listed above
        2. Generate queries that are COMPLETELY DIFFERENT from all previous attempts
        
        Please generate {NUM_SEARCH_QUERIES} COMPLETELY NEW search queries following the strategy above.
        
        You must respond with a list of queries in the following format:
        [
            {{"search_query": "query1", "user_question": "{user_question}" }},
            {{"search_query": "query2", "user_question": "{user_question}" }},
            {{"search_query": "query3", "user_question": "{user_question}" }}
        ]
        """
    
    # 2- Get the LLM response
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content
    
    # 3- Parse the response to get the search queries
    try:
        # Extract the JSON array from the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        json_str = response_text[json_start:json_end]
        
        # Parse the JSON
        search_queries = json.loads(json_str)
        
        print(f"Generated {len(search_queries)} search queries")
        for i, query in enumerate(search_queries):
            print(f"  Query {i+1}: {query['search_query']}")
        
        # 4- Return the updated state
        return {
            "search_queries": search_queries,
            # Reset the relevance evaluation and regeneration flag when generating new queries
            "relevance_evaluation": None,
            "should_regenerate_queries": None
        }
    except Exception as e:
        print(f"Error parsing search queries: {str(e)}")
        # Fallback to a default search query if parsing fails
        default_queries = [
            {"search_query": f"{user_question} iteration {iteration_count + 1}", "user_question": user_question}
        ]
        print(f"Using default query: {default_queries[0]['search_query']}")
        return {
            "search_queries": default_queries,
            "relevance_evaluation": None,
            "should_regenerate_queries": None
        }

def SubAgentPerformWebSearches(state: Dict[str, Any]) -> Dict[str, Any]:
    # Web Searcher Agent Node Function (Component): Conducts searches and gathers 
    # URLs based on the generated queries
    NUM_SEARCH_RESULTS_PER_QUERY = 3
    """
    Perform web searches based on the generated search queries.
    """
    search_queries = state["search_queries"]
    search_results = []
    fallback_used = False
    
    print(f"Performing web searches for {len(search_queries)} queries...")
    
    # 1- For each search query, get the search results
    for query_obj in search_queries:
        search_query = query_obj["search_query"]
        user_question = query_obj["user_question"]
        
        try:
            # 1-1 Get the search results
            print(f"Searching for: {search_query}")
            urls = websearch_for_agent_with_duckDuckGoSearchAPIWrapper(web_query=search_query, num_results=NUM_SEARCH_RESULTS_PER_QUERY)
            
            # 1-2 Check if these are likely fallback results (Wikipedia URLs)
            if any("wikipedia.org" in url for url in urls[:2]):
                print(f"Fallback search was used for query: {search_query}")
                fallback_used = True
                is_fallback = True
            else:
                is_fallback = False
            
            # 1-3 Add the results to the list
            for url in urls:
                search_results.append({
                    "result_url": url,
                    "search_query": search_query,
                    "user_question": user_question,
                    "is_fallback": is_fallback
                })
                
            print(f"Found {len(urls)} results for query: {search_query}")
        except Exception as e:
            print(f"Error searching for '{search_query}': {str(e)}")
            # Continue with other queries even if one fails
            continue
    
    # 2- If we have no search results at all, add a fallback result
    if not search_results:
        print("No search results found. Using general fallback information.")
        fallback_url = "https://en.wikipedia.org/wiki/Main_Page"
        search_results.append({
            "result_url": fallback_url,
            "search_query": "general information",
            "user_question": state["user_question"],
            "is_fallback": True
        })
        fallback_used = True
    
    # 3- Return the updated state with information about fallback usage
    return {
        "search_results": search_results,
        "used_fallback_search": fallback_used
        }

def SubAgentSummarizeSearchResults(state: Dict[str, Any]) -> Dict[str, Any]:
        # Content Summarizer Agent Node Function (Component): Scrapes and summarizes 
        # the content of web pages
        RESULT_TEXT_MAX_CHARACTERS = 10000
        """
        Summarize the search results.
        """
        search_results = state["search_results"]
        used_fallback_search = state.get("used_fallback_search", False)
        summaries = []
        
        print(f"Summarizing {len(search_results)} search results...")
        
        # 1- For each search result, get the text and summarize it
        for result in search_results:
            result_url = result["result_url"]
            search_query = result["search_query"]
            user_question = result["user_question"]
            is_fallback = result.get("is_fallback", False)
            
            try:
                # 1-1 Get the webpage content
                print(f"Scraping content from: {result_url}")
                search_result_text = web_scrape(url=result_url)[:RESULT_TEXT_MAX_CHARACTERS]
                
                # 1-2 Skip if the content is an error message or too short
                if search_result_text.startswith("Failed to") or len(search_result_text) < 50:
                    print(f"Skipping {result_url} due to scraping issues or insufficient content")
                    continue
                
                # 1-3 Format the prompt, with additional context for fallback results
                if is_fallback:
                    prompt = f"""
                    You are summarizing content from a fallback source that was used because the primary search engine was unavailable.
                    
                    Read the following text:
                    Text: {search_result_text} 
                    
                    -----------
                    
                    Using the above text, answer in short the following question.
                    Question: {search_query}
                    
                    -----------
                    If you cannot answer the question above using the text provided above, then just summarize the text. 
                    Include all factual information, numbers, stats etc if available.
                    
                    Note that this is a fallback source, so it might not directly address the question.
                    """
                else:
                    prompt = SUMMARY_PROMPT_TEMPLATE.format(
                        search_result_text=search_result_text,
                        search_query=search_query
                    )
                
                # 1-4 Get the summary
                summary_response = llm.invoke(prompt)
                text_summary = summary_response.content
                
                # 1-5 Add a note about fallback sources
                if is_fallback:
                    source_note = "[Note: This information comes from a fallback source and may not directly address the question.]"
                    text_summary = f"{text_summary}\n{source_note}"
                
                # 1-6 Create the summary object
                summary = {
                    "summary": f"Source Url: {result_url}\nSummary: {text_summary}",
                    "result_url": result_url,
                    "user_question": user_question,
                    "is_fallback": is_fallback
                }
                
                summaries.append(summary)
                print(f"Successfully summarized content from: {result_url}")
            except Exception as e:
                print(f"Error summarizing {result_url}: {str(e)}")
                # Skip this result if there's an error
                continue
        
        # 2- Create the research summary
        if summaries:
            research_summary = "\n\n".join([s["summary"] for s in summaries])
            print(f"Created research summary with {len(summaries)} sources")
            
            # 2-1 Add a note if fallback search was used
            if used_fallback_search:
                fallback_note = "\n\n[Note: Some or all of this information comes from fallback sources because the primary search engine was unavailable. The information may not be as directly relevant to your question as usual.]"
                research_summary += fallback_note
        else:
            research_summary = "No relevant information found. Please try different search queries."
            print("Warning: No summaries were generated from search results")
        
        # 3- Return the updated state
        return {
            "search_summaries": summaries,
            "research_summary": research_summary,
            "used_fallback_search": used_fallback_search
        }

def SubAgentEvaluateSearchRelevance(state: Dict[str, Any]) -> Dict[str, Any]:
    # Relevance Evaluator Agent Node Function (Component): Assesses if the summaries are relevant 
    # enough to proceed or if new search queries are needed
    """
    Evaluate the relevance of search summaries to the original question.
    If less than 50% of summaries are relevant, return to search query generation.
    """
    search_summaries = state.get("search_summaries", [])
    user_question = state["user_question"]
    research_summary = state.get("research_summary", "")
    used_fallback_search = state.get("used_fallback_search", False)
    
    print("Evaluating relevance of search summaries to the original question...")
    
    # 1- If there are no summaries, we need to regenerate queries
    if not search_summaries or not research_summary:
        print("No search summaries found. Regenerating search queries...")
        return {"should_regenerate_queries": True}
    
    # 2- Use the LLM to evaluate relevance
    
    # 3- Create a prompt for the LLM to evaluate relevance
    evaluation_prompt = f"""
    You are an expert research evaluator. Your task is to evaluate the relevance of search results 
    to the original research question.
    
    Original research question: {user_question}
    
    Search result summaries:
    {research_summary}
    
    For each search result summary, determine if it is relevant to answering the original question.
    Then calculate what percentage of the search results are relevant.
    
    Return your evaluation as a JSON object with the following structure:
    {{
        "relevance_percentage": <percentage of relevant results as a number between 0 and 100>,
        "explanation": <brief explanation of your evaluation>,
        "relevant_count": <number of relevant summaries>,
        "total_count": <total number of summaries>
    }}
    """
    
    try:
        # 4- Get the evaluation from the LLM
        evaluation_response = llm.invoke(evaluation_prompt)
        evaluation_text = evaluation_response.content
        
        # 5- Extract the JSON from the response
        try:
            # 5-1 Find JSON in the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            json_str = evaluation_text[json_start:json_end]
            
            # 5-2 Parse the JSON
            evaluation = json.loads(json_str)
            relevance_percentage = evaluation.get("relevance_percentage", 0)
            
            # 5-3 Determine if we should regenerate queries (less than 50% relevant)
            should_regenerate = relevance_percentage < 50
            
            if should_regenerate:
                print(f"Only {relevance_percentage}% of search results are relevant. Regenerating search queries...")
            else:
                print(f"{relevance_percentage}% of search results are relevant. Proceeding to write research report...")
            
            return {
                "relevance_evaluation": evaluation,
                "should_regenerate_queries": should_regenerate
            }
        except Exception as e:
            print(f"Error parsing relevance evaluation: {str(e)}")
            # If we can't parse the evaluation, assume we need to regenerate
            return {"should_regenerate_queries": True}
    except Exception as e:
        print(f"Error during relevance evaluation: {str(e)}")
        # If there's an error, assume we need to regenerate
        return {"should_regenerate_queries": True}

#endregion

#region Settings
def get_llm():
    openai_api_key = get_env_api_key ("OpenAI")
    llm_model = get_llm_model("GPT-Cheapest")
    llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)
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
    
#region Utils
# Create a singleton instance of the DuckDuckGoSearchAPIWrapper to reuse
_duckDuckGoSearch_singleton_instance = None
# Track the last request time to implement rate limiting
_last_request_time_for_rate_limiting = 0
_min_request_interval = 2.0  # Minimum seconds between requests

def websearch_get_ddg_instance_for_agent_with_duckDuckGoSearchAPIWrapper():
    """Get a singleton instance of DuckDuckGoSearchAPIWrapper."""
    global _duckDuckGoSearch_singleton_instance
    if _duckDuckGoSearch_singleton_instance is None:
        _duckDuckGoSearch_singleton_instance = DuckDuckGoSearchAPIWrapper()
    return _duckDuckGoSearch_singleton_instance

def websearch_for_agent_with_duckDuckGoSearchAPIWrapper(web_query: str, num_results: int) -> List[str]:

    # from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    # from duckduckgo_search.exceptions import DuckDuckGoSearchException => # pip install -U duckduckgo-search

    # import time
    # import random

    """
    Perform a web search with rate limiting, retry logic, and fallback mechanisms.
    
    Args:
        web_query: The search query
        num_results: Number of results to return
    
    Returns:
        List of URLs from search results
    """
    global _last_request_time_for_rate_limiting
    
    # Implement rate limiting
    current_time = time.time()
    time_since_last_request = current_time - _last_request_time_for_rate_limiting
    
    if time_since_last_request < _min_request_interval:
        # Wait to avoid hitting rate limits
        sleep_time = _min_request_interval - time_since_last_request + random.uniform(0.1, 0.5)
        print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next search request...")
        time.sleep(sleep_time)
    
    # Try DuckDuckGo with retry logic
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Update the last request time
            _last_request_time_for_rate_limiting = time.time()
            
            # Get the search results
            ddg = websearch_get_ddg_instance_for_agent_with_duckDuckGoSearchAPIWrapper()
            results = ddg.results(web_query, num_results)
            
            # Extract the URLs
            urls = [r["link"] for r in results]
            
            # If we got results, return them
            if urls:
                return urls
            else:
                print(f"No results found for query: {web_query}")
                # If no results, try a fallback on the last attempt
                if attempt == max_retries - 1:
                    break
                time.sleep(1)  # Brief pause before retrying
                
        except DuckDuckGoSearchException as e:
            if "Ratelimit" in str(e) and attempt < max_retries - 1:
                # Exponential backoff with jitter for rate limit errors
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"DuckDuckGo rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"DuckDuckGo search failed: {str(e)}")
                break
        except Exception as e:
            print(f"Error during web search: {str(e)}")
            if attempt < max_retries - 1:
                # Simple retry for other errors
                delay = base_delay + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                break
    
    # If we get here, all attempts failed or returned no results
    # Use a fallback search mechanism
    return websearch_fallback_search_for_agent_with_duckDuckGoSearchAPIWrapper(web_query, num_results)

def websearch_fallback_search_for_agent_with_duckDuckGoSearchAPIWrapper(query: str, num_results: int) -> List[str]:
    """
    Fallback search mechanism when DuckDuckGo is rate-limited or fails.
    Returns a list of relevant Wikipedia and general knowledge URLs.
    
    Args:
        query: The search query
        num_results: Number of results to return
    
    Returns:
        List of URLs that might be relevant to the query
    """
    print(f"Using fallback search for query: {query}")
    
    # Clean and prepare the query
    query_clean = query.lower().strip()
    
    # List of general knowledge sources that cover a wide range of topics
    general_sources = [
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://www.britannica.com/",
        "https://www.worldcat.org/",
        "https://www.jstor.org/",
        "https://www.sciencedirect.com/",
        "https://www.researchgate.net/",
        "https://scholar.google.com/",
        "https://www.academia.edu/"
    ]
    
    # Try to generate Wikipedia URLs based on key terms in the query
    wikipedia_urls = []
    
    # Extract potential Wikipedia topics from the query
    # Remove common question words and stop words
    stop_words = ["what", "where", "when", "why", "how", "is", "are", "was", "were", 
                 "do", "does", "did", "can", "could", "would", "should", "might",
                 "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
                 "to", "of", "from", "as", "tell", "me", "about", "you", "i", "we"]
    
    # Split the query into words and filter out stop words
    query_words = [word for word in query_clean.split() if word not in stop_words]
    
    # Generate potential Wikipedia URLs
    if len(query_words) >= 2:
        # Try pairs of words
        for i in range(len(query_words) - 1):
            topic = "_".join([query_words[i].capitalize(), query_words[i+1].capitalize()])
            wikipedia_urls.append(f"https://en.wikipedia.org/wiki/{topic}")
    
    # Add single word topics
    for word in query_words:
        if len(word) > 3:  # Only use meaningful words
            wikipedia_urls.append(f"https://en.wikipedia.org/wiki/{word.capitalize()}")
    
    # Try to create a direct Wikipedia search URL
    search_query = query_clean.replace(" ", "+")
    wikipedia_search_url = f"https://en.wikipedia.org/w/index.php?search={search_query}"
    wikipedia_urls.insert(0, wikipedia_search_url)
    
    # Combine Wikipedia URLs with general sources
    all_urls = wikipedia_urls + general_sources
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in all_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    # Return the top N results
    return unique_urls[:num_results]
        
def web_scrape(url: str) -> str:
    # import requests
    # from bs4 import BeautifulSoup    
    # We scrape the web pages from the result list using Beautiful Soup, which is a web scraper library.
    try:
        # Add request headers; otherwise,Wikipedia will block the call.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)

            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

def websearch_with_duckDuckGoSearchAPIWrapper(web_query: str, num_results: int) -> List[str]:
     # from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    # from typing import List

    # NOTE Other web search engine wrappers provided by LangChain are TavilySearchResults 
    # and GoogleSearchAPIWrapper. 
    # Both require an API key, so we chose DuckDuckGoSearchAPIWrapper because it doesn’t.

    # use the LangChain wrapper for the DuckDuckGo search engine to perform web
    # searches. Its results method returns a list of objects, each containing the result URL in
    # a property called "link".    
    return [
        r["link"] 
        for r in DuckDuckGoSearchAPIWrapper().results(
            web_query, num_results)
    ] 

def json_parser (json_text):
    # import json  
    # from langchain_core.runnables import RunnableLambda

    # A utility function that converts JSON text from the LLM into a Python object,
    # which is usually a dictionary or sometimes a list. If the JSON is malformed, 
    # it will return an empty dictionary.

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {}        
#endregion   

#region main
openai_api_key = get_env_api_key ("OpenAI")
llm_model = get_llm_model("GPT-Cheapest")
llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)

if __name__ == "__main__":
    # STEP 5: Run the AI Agent Implemented with LangGraph (Agentic workflows with LangGraph) 
    # For testing purposes

    # Example usage
    question = "What can you tell me about the roles of C# Data Types and EF Core data types in implementing web applications?"
    report = run_research(question)
    print(report)
    ## Result:  
# Generating initial search queries...
# Generated 3 search queries
#   Query 1: C# data types value types reference types nullability and performance in web applications; EF Core data type mappings and database type configurations
#   Query 2: EF Core data type mappings to SQL databases: how C# types map to database column types, configuration, migrations, and nullability
#   Query 3: Best practices selecting C# and EF Core data types for web apps: value vs reference types, nullability, performance, and common pitfalls in mappings
# Performing web searches for 3 queries...
# Searching for: C# data types value types reference types nullability and performance in web applications; EF Core data type mappings and database type configurations
# Found 3 results for query: C# data types value types reference types nullability and performance in web applications; EF Core data type mappings and database type configurations
# Searching for: EF Core data type mappings to SQL databases: how C# types map to database column types, configuration, migrations, and nullability
# Found 3 results for query: EF Core data type mappings to SQL databases: how C# types map to database column types, configuration, migrations, and nullability
# Searching for: Best practices selecting C# and EF Core data types for web apps: value vs reference types, nullability, performance, and common pitfalls in mappings
# Rate limiting: Waiting 0.48 seconds before next search request...
# Found 3 results for query: Best practices selecting C# and EF Core data types for web apps: value vs reference types, nullability, performance, and common pitfalls in mappings
# Summarizing 9 search results...
# Scraping content from: https://www.bookey.app/book/entity-framework-core-in-action
# Successfully summarized content from: https://www.bookey.app/book/entity-framework-core-in-action
# Scraping content from: https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/types
# Successfully summarized content from: https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/types
# Scraping content from: https://www.damirscorner.com/blog/posts/20220812-NullableReferenceTypesAndNullabilityInEfCore.html
# Successfully summarized content from: https://www.damirscorner.com/blog/posts/20220812-NullableReferenceTypesAndNullabilityInEfCore.htmlScraping content from: https://cdn.bookey.app/files/pdf/book/en/entity-framework-core-in-action.pdf
# Successfully summarized content from: https://cdn.bookey.app/files/pdf/book/en/entity-framework-core-in-action.pdf
# Scraping content from: https://github.com/dotnet/efcore
# Successfully summarized content from: https://github.com/dotnet/efcore
# Scraping content from: https://www.udemy.com/course/introduction-to-entity-framework-core/
# Successfully summarized content from: https://www.udemy.com/course/introduction-to-entity-framework-core/
# Scraping content from: https://github.com/dotnet/efcore
# Successfully summarized content from: https://github.com/dotnet/efcore
# Scraping content from: https://www.youtube.com/watch?v=vLnPwxZdW4Y
# Successfully summarized content from: https://www.youtube.com/watch?v=vLnPwxZdW4Y
# Scraping content from: https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/builtin-types/arrays
# Successfully summarized content from: https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/builtin-types/arrays
# Created research summary with 9 sources
# Evaluating relevance of search summaries to the original question...
# 50% of search results are relevant. Proceeding to write research report...
# Iteration 1: Search results are relevant. Proceeding to write report.
# Roles of C# Data Types and EF Core Data Types in Implementing Web Applications: A Structured Analysis Based on Available Sources

# Abstract
# This report analyzes the roles that C# data types and EF Core data types play in building web applications, drawing on a curated set of sources that describe C# type semantics, the architecture and capabilities of EF Core, and the interaction between C# nullability features and EF Core migrations. Because the available materials do not provide a canonical mapping of C# types
# to database column types, the analysis foregrounds foundational concepts (value vs. reference types, nullability, arrays, boxing) and high-level EF Core capabilities (object-relational mapping, DbContext, change tracking, migrations, and provider-based database access). The report also highlights practical workflows and design considerations when integrating C# type semantics with EF Core in web projects. Where relevant, it notes gaps and points to authoritative documentation for detailed mappings and configurations.

# Introduction
# Web applications built on the .NET stack rely on a precise alignment between in-memory data representations (as defined by the C# type system) and persisted data representations (as defined by database schemas via EF Core). The C# language defines a rich type system that distinguishes value types from reference types, governs nullability (including nullable reference types introduced in recent language revisions), and determines how data flows through memory (including boxing/unboxing semantics and
# default constructors). EF Core, as a modern object-relational mapper (O/RM), translates domain models into relational data structures, tracks changes, and coordinates migrations and queries across multiple database providers. The interplay between C# types and EF Core data types shapes every aspect of web application development—from data modeling and validation to performance considerations and maintainability.

# This analysis synthesizes information from several sources that collectively illuminate (a) the roles and semantics of C# data types, (b) the architectural features and capabilities of EF Core, and (c) how nullable reference types (NRT) influence EF Core’s nullability handling and migrations. While the sources do not provide a comprehensive mapping of C# types to specific SQL types (and explicitly note that such mappings are not covered in some discussions), they collectively offer a detailed view of how developers should reason about data types and EF Core in web app development.

# Part 1: Core Roles of C# Data Types in Web Applications

# 1. Value vs. Reference Types: Semantics and Implications
# C# categorizes types into value types (structs and enums) and reference types (classes, interfaces, arrays, delegates, etc.).
# Value types directly contain their data, whereas reference types store a reference to data that may be shared by multiple variables. In web applications, this distinction matters for memory usage, performance, and mutation semantics:
# - Value types: Copy semantics are straightforward (assignment copies the data). They can lead to lower-level, stack-allocated
# storage in tight loops and for small data, which can be beneficial for per-request processing pipelines, serialization, and performance-critical calculations.
# - Reference types: Copy semantics copy references, enabling shared access to a single object instance. This is typical for domain models, services, and repositories where identity and shared state matter. However, inadvertent sharing can introduce unintended side effects if multiple components mutate the same object.

# These fundamental behaviors influence how entities and DTOs are designed in EF Core-based web apps. The C# type system’s distinction underpins how EF Core materializes entities, tracks changes, and computes diffs for persistence.

# 2. Nullability: From Reference Types to Nullable Reference Types
# Nullability is central to data integrity and user-facing validation. Historically, reference type properties could be null by
# default, complicating database schema decisions and runtime checks. The introduction and adoption of nullable reference types
# (NRT) in C# provide a compiler-enforced mechanism to annotate and enforce nullability expectations, which has downstream effects for EF Core migrations:
# - Pre-NRT: Reference properties were nullable in the database by default. Developers often used [Required] data annotations to enforce non-null columns, with migrations reflecting those constraints.
# - Post-NRT (Nullable enable): The non-nullability of a C# reference property becomes the determinant of the corresponding database column’s NULL/NOT NULL status. If a string property is declared as string (non-nullable in code) it tends toward NOT NULL in the database; string? becomes NULLABLE. This shifts nullability decisions from the data annotations alone to the actual C# type annotations.
# - Practical workflow: Enabling NRT requires re-evaluating existing migrations. Developers may need to adjust code to maintain
# the desired database schema (e.g., adding default values or constructor initialization to satisfy compiler warnings CS8618). The workflow often includes generating a migration to verify that no unintended changes occur, potentially iterating on code to produce an empty migration, and then finalizing a stable schema.

# These aspects emphasize a design principle: EF Core’s nullability must be aligned with C#’s nullability semantics to avoid data-loss risks and to ensure predictable migrations. The Damirscorner article provides practical guidance on this interplay, including how to manage migrations and warnings during the transition to NRT.

# 3. Boxing/Unboxing and Boxing Overheads
# Boxing/unboxing is a concern when value types (e.g., int, struct types) are treated as objects. In web applications, boxing overhead can appear in data access layers or in situations where value types are used in non-generic collections or when APIs expect object types. While not specific to EF Core, awareness of boxing implications is relevant when designing data transfer objects or dynamic data flows (e.g., dynamic LINQ queries or heterogeneous collections). The cited material notes boxing/unboxing as a core aspect of the value/reference type distinction.

# 4. Arrays, Initialization, and Collection Semantics
# Arrays are a fundamental collection type in C#, and they are reference types themselves. They can be nullable, and their elements can be value or reference types. For web applications, arrays often appear in:
# - Data transfer scenarios where fixed-size or batch data structures are serialized to JSON or other payloads.
# - Query and paging scenarios where arrays or array-like structures are manipulated in memory before being persisted or returned to clients.
# Understanding array semantics (nullable array vs. non-nullable element types, default element values, and 0-based indexing) helps in designing endpoints and services that reliably serialize/deserialize data. The arrays reference material also highlights nuances such as:
# - Default initialization: elements of reference type arrays default to null, while value types default to their zero-value (e.g., 0 for int).
# - Initialization forms: collection expressions and initializers enable concise construction of arrays and collections, affecting readability and maintainability of code in controllers and service layers.

# These details, while not EF Core-specific, influence how web applications prepare data for persistence and transmission, including model binding and serialization concerns.

# Section 1 Takeaways for Web App Design
# - Use value types for performance-sensitive data (e.g., numeric calculations in domain logic) where copying semantics are acceptable and predictable.
# - Leverage reference types for domain entities and services where identity, mutability, and shared state are central, while being mindful of unintended side effects from shared references.
# - Adopt nullable reference types to tighten null-safety guarantees, but plan migrations carefully to avoid unintended database changes.
# - Be mindful of boxing overhead when value types are used in object-requiring contexts, especially in data access and serialization layers.
# - Use arrays and collection types thoughtfully in data models and DTOs, ensuring consistent initialization and nullability expectations to minimize runtime errors.

# Part 2: Core Roles of EF Core Data Types in Web Applications

# 1. EF Core as an Object-Relational Mapper
# EF Core positions itself as a modern, cross-platform O/RM for .NET, providing a bridge between in-memory domain models and relational databases (with NoSQL considerations in some contexts). Its primary roles include:
# - Mapping .NET classes to relational tables, and properties to columns, enabling developers to work with strongly-typed domain models rather than raw SQL.
# - Querying data through LINQ, where EF Core translates expressions into SQL queries executed by the underlying provider.
# - Change tracking and update propagation: EF Core tracks entity state (added, modified, deleted) and generates appropriate SQL statements to synchronize the database with in-memory changes.
# - Migrations and database creation/update workflows: EF Core provides code-based migrations to evolve the database schema in step with model changes, along with features to create and update databases from the model.

# These capabilities are foundational for maintainable, testable, and scalable web applications. They enable developers to express business logic in domain terms while delegating the persistence concerns to a robust ORM framework.

# 2. Architecture and Internal Model
# EF Core’s architecture centers on the DbContext and the modeling pipeline:
# - DbContext: The primary unit of work and a gateway to querying and persisting data. It tracks entities, coordinates queries,
# and orchestrates save operations.
# - Modeling and internal data model: EF Core uses a detailed internal representation of the domain model, including entity shape, relationships, keys, and constraints, to drive SQL generation and migration behavior.
# - LINQ-based querying and change tracking: The combination of LINQ expressions and the context’s change tracker enables expressive queries and accurate update statements, minimizing boilerplate SQL in application code.

# This architectural overview underscores why C# types matter: the shape and semantics of our C# domain models (classes, properties, and their types) are what EF Core maps to database structures, how it enforces constraints, and how it detects changes during SaveChanges.

# 3. Database-Agnostic Fundamentals and Providers
# EF Core is provider-based: a core abstraction with specific database providers (e.g., SQL Server, SQLite, Cosmos DB, PostgreSQL, MySQL) implementing the details of SQL dialects, data type mappings, and provider-specific capabilities. The presence of multiple providers demonstrates EF Core’s cross-database flexibility, while also implying a dependency on consistent domain modeling across providers:
# - Basic usage: Creating a DbContext, adding entities, querying with LINQ, updating, and saving changes.
# - Provider ecosystem and docs: The official EF Core repository and docs enumerate the supported providers and configuration options, enabling developers to switch or exper

#endregion