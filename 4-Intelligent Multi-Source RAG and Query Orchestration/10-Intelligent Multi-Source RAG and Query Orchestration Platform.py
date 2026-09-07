import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
import json

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import List

import requests
from bs4 import BeautifulSoup

import re

from langchain_community.document_loaders import WikipediaLoader, Docx2txtLoader, PyPDFLoader, TextLoader

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_classic.storage import InMemoryByteStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
import uuid

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from langchain_core.output_parsers import BaseOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

from langchain_classic.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator

from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from typing import Literal, Optional, Tuple, List


#region Settings
def get_llm_with_structured_output(question):
    openai_api_key = get_env_api_key ("OpenAI")
    llm_model = get_llm_model("GPT-Cheapest")
    llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model).with_structured_output(question)
    return llm
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
        
def reciprocal_rank_fusion(results_groups: list[list], k=60):
    # RRF ALGORITHM:
    # The core of this workflow is the RRF algorithm, which assigns scores to documents
    # retrieved by multiple queries. Using the RRF formula, each document is scored based
    # on its rank and then reranked by total RRF score. See the following listing for implementation
    # details.

    # Based on: https://github.com/Raudaschl/rag-fusion/blob/master/main.py     
    """ Reciprocal_rank_fusion that takes multiple groups of 
        ranked documents and an optional parameter k used in 
        the Reciprocal Rank Fusion (RRF) formula """

    indexed_results = {} # Initialize a dictionary to organize results with an index
    
    for group_id, results_group in enumerate(results_groups): # Index the results by (group_id, local_rank)
        for local_rank, doc in enumerate(results_group):
            indexed_results[(group_id, local_rank)] = doc
    
    fused_scores = {} # Initialize a dictionary to hold fused scores for each unique document
    
    for key, doc in indexed_results.items(): # Iterate through the indexed results
        group_id, local_rank = key

        if key not in fused_scores:
            fused_scores[key] = 0 # Initialize an indexed result with a score of 0 if it has not been processed yet
        
        doc_current_score = fused_scores[key]        
        fused_scores[key] += 1 / (local_rank + k) # calculate the new document score with the RRF formula

    reranked_results = [ # rerank the results by RRF score
        (indexed_results[key], score)
        for key, score in sorted(fused_scores.items(), 
                                key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) 

def execute_rag_chain(question, chosen_retriever):
    full_rag_chain = (
        {
            "context": {"question": RunnablePassthrough()} 
                | chosen_retriever,# The context is returned by the retriver after feeding to it the rewritten query
            "question": RunnablePassthrough(),# This is the original user question
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return full_rag_chain.invoke(question) 

def retriever_chooser(question):
    selected_data_source = question_router.invoke(
        {"question": question})

    return retriever_chains[selected_data_source.datasource]

class RouteQuery(BaseModel):
    """Route a user question to the most relevant datasource."""

    datasource: Literal["tourism_info_store", 
        "uk_booking_db"] = Field(
        ...,
        description="""Given a user question, 
        route it either to a tourism info vector store 
        or a UK accomodation booking relational database.""",
    )

class DestinationSearch(BaseModel):
    # The DestinationSearch class translates the user question into a structured object with
    # a content_search field containing the question (minus filtering details) and fields for
    # inferred search filters.

    # Strongly typed structured question    
    """Search over a vector database of tourism destinations."""

    content_search: str = Field(
        "",
        description="""Similarity search query applied 
        to tourism destinations.""",
    )
    destination: str = Field(
        ...,
        description="The specific UK destination to be searched.",
    )
    region: str = Field(
        ...,
        description="The name of the UK region to be searched.",
    )

def build_filter(destination_search: DestinationSearch):
    # BUILDING A CHROMADB FILTER STATEMENT FROM THE STRUCTURED QUERY
    # create a function to convert a DestinationSearch object into a filter compatible with ChromaDB
    comparisons = []

    destination = destination_search.destination # Get destination and region from the structured query
    region = destination_search.region # Get destination and region from the structured query
    
    if destination and destination != '': # If the destination exists, create an 'equality' operation
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="destination",
                value=destination,
            )
        )
    if region and region != '': # If the region exists, create an 'equality' operation
        comparisons.append(
            Comparison(
                comparator=Comparator.EQ,
                attribute="region",
                value=region,
            )
        )    

    search_filter = Operation(operator=Operator.AND, 
                            arguments=comparisons) # Create a combined search filter

    chroma_filter = ChromaTranslator().visit_operation(
        search_filter) # Transform the filter into Chroma format
        
    return chroma_filter

def pretty_print(self) -> None:
    for field in self.__fields__:
        if getattr(self, field) is not None and getattr(
            self, field) != getattr(
            self.__fields__[field], "default", None
        ):
            print(f"{field}: {getattr(self, field)}")

def split_html_docs_into_chunks(docs, html2text_transformer, text_splitter):
    # define a function to split the content into coarse chunks
    text_docs = html2text_transformer.transform_documents(docs) # transform HTML docs into clean text docs 
    chunks = text_splitter.split_documents(text_docs)

    return chunks

def load_html_content_with_asyncHtmlLoader(urlAddress):
    # from langchain_community.document_loaders import AsyncHtmlLoader

    # Loading the HTML content with the AsyncHtmlLoader
    # The final step is to ingest some content about Cornwall, a region in the UK known for
    # its stunning seaside resorts, using an HTML loader:
    destination_url = urlAddress
    html_loader = AsyncHtmlLoader(destination_url)
    docs = html_loader.load()
    len(docs)
    return docs
    # This snippet fetches the Cornwall page content, which we’ll use to create both granular
    # and coarse chunks.

def create_chormaDb_collection_and_reset(chroma_collection_name, openai_api_key):
    # This will initialize a new Chroma collection called cornwall_granular_collection. If
    # the collection already exists, it will be reset to start fresh.    
    collection = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
        )
    collection.reset_collection() # Reset the collection in case it already exists
    return collection

#endregion
    
openai_api_key = get_env_api_key ("OpenAI")
llm_model = get_llm_model("GPT-Cheapest")
llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)

#region main 1: Self metadata query

## Step 1-1: Ingestion (metadata enriched)
# DEFINING METADATA
# Identify keywords to tag each chunk, such as the following:
#    source—URL of the original content
#    destination—The tourism destination referenced
#    region—The UK region of the destination
# Manually define mappings for destination and region, and then dynamically generate
# the source URL for each chunk.

##Ingestion (metadata enriched)
uk_with_metadata_collection = create_chormaDb_collection_and_reset("uk_child_chunks", openai_api_key)

###Defining content to be ingested and splitting strategy
html2text_transformer = Html2TextTransformer()

text_splitter = RecursiveCharacterTextSplitter( # Instantiate a relatively fine-chunk splitting strategy
    chunk_size=1000, chunk_overlap=100
)
uk_destinations = [
    ("Cornwall", "Cornwall"), ("North_Cornwall", "Cornwall"), 
    ("South_Cornwall", "Cornwall"), ("West_Cornwall", "Cornwall"),
    ("Tintagel", "Cornwall"), ("Bodmin", "Cornwall"), 
    ("Wadebridge", "Cornwall"),
    ("Penzance", "Cornwall"), ("Newquay", "Cornwall"), 
    ("St_Ives", "Cornwall"),
    ("Port_Isaac", "Cornwall"), ("Looe", "Cornwall"), 
    ("Polperro", "Cornwall"),
    ("Porthleven", "Cornwall"),
    ("East_Sussex", "East_Sussex"), ("Brighton", "East_Sussex"),
    ("Battle", "East_Sussex"), ("Hastings_(England)", "East_Sussex"),
    ("Rye_(England)", "East_Sussex"), ("Seaford", "East_Sussex"), 
    ("Ashdown_Forest", "East_Sussex")
]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_url_with_metadata = [ # Prepare metadata to be imported: Url, UK Destination and UK Region 
    ( f'{wikivoyage_root_url}/{destination}', destination, region)
    for destination, region in uk_destinations]

###INGESTING CONTENT WITH METADATA
### Enriching a document with metadata: updating metadata
tintagel_url, tintagel_destination, tintagel_region = uk_destination_url_with_metadata[4]

tintagel_docs = load_html_content_with_asyncHtmlLoader(tintagel_url)

# tintagel_docs # COMMENT: LangChain loaders create docs which contain metadata

for doc in tintagel_docs:
    doc.metadata['destination'] = tintagel_destination
    doc.metadata['region'] = tintagel_region
    print(doc.metadata)

###Enriching a document with metadata: creating metadata 
tintagel_docs_with_metadata = [
    Document(page_content=d.page_content,
             metadata = {
                 'source': tintagel_url,
                 'destination': tintagel_destination,
                 'region': tintagel_region
             })
    for d in tintagel_docs
]
# tintagel_docs_with_metadata # examine the Document

###Enriching the UK destination documents with metadata: creating metadata 
# Enrich the content with metadata by processing each document chunk
for (url, destination, region) in uk_destination_url_with_metadata:
    html_loader = AsyncHtmlLoader(url) # Loader for one destination
    docs =  html_loader.load() # Documents (chunks) related to one destination
    
    docs_with_metadata = [
        Document(page_content=d.page_content,
        metadata = {
            'source': url,
            'destination': destination,
            'region': region})
        for d in docs]
             
    chunks = split_html_docs_into_chunks(docs_with_metadata, html2text_transformer, text_splitter)

    print(f'Importing: {destination}')
    uk_with_metadata_collection.add_documents(documents=chunks) 

# Now our collection is ready, with each document chunk enriched with metadata. we
# can query this content and apply metadata filters to refine search results based on keywords
# such as destination, region, or source.

## Step 1-2: Q & A on a collection enriched with metadata
### Step 1-2-1: Searching the collection with a metadata filter explicitly
# There are three ways to query metadata-enriched content:
#    Explicit metadata filters—Specify the metadata filter manually.
#    SelfQueryRetriever—Automatically generate the metadata filter using the
#     SelfQueryRetriever.
#    Structured LLM function call—Infer the metadata filter with a structured call to
#     an LLM function.

# QUERYING WITH AN EXPLICIT METADATA FILTER:
# use the metadata attached to each chunk by explicitly adding a filter to the retriever.

question =  "Events or festivals"
metadata_retriever = uk_with_metadata_collection.as_retriever(
    search_kwargs={'k':2, 'filter':{'destination': 'Newquay'}})

result_docs = metadata_retriever.invoke(question)
result_docs
# COMMENT: As we can see, only chunks associated with'destination': 'Newquay' have been selected

# To adjust the filter, instantiate a new retriever with the updated parameters.


### Step 1-2-2: Generating the self metadata query with the SelfQueryRetriever
# from langchain_classic.chains.query_constructor.base import AttributeInfo 
# from langchain_classic.retrievers.self_query.base import SelfQueryRetriever # this requires pip install lark

# AUTOMATICALLY GENERATING METADATA FILTERS WITH SELFQUERYRETRIEVER:

# we can also generate metadata filters automatically with SelfQueryRetriever. This
# tool interprets the user’s question to infer the appropriate filter criteria. The underlying
# engine that performs this inference is, of course, the LLM—meaning this
# approach introduces additional cost and latency.

# define the metadata attributes to infer from the question:
metadata_field_info = [
    AttributeInfo(
        name="destination",
        description="The specific UK destination to be searched",
        type="string",
    ),
    AttributeInfo(
        name="region",
        description="The name of the UK region to be searched",
        type="string",
    )
]
# set up the SelfQueryRetriever with the question, without specifying a manual filter
question = "Tell me about events or festivals in the UK town of Newquay"

self_query_retriever = SelfQueryRetriever.from_llm(
    llm, uk_with_metadata_collection, question, 
    metadata_field_info, verbose=True
)
# Invoke the retriever with the question
result_docs = self_query_retriever.invoke(question)

### Step 1-2-3: Generating the self metadata query with a LLM function call
#### Step 1-2-3-1: Query schema
# we can also infer metadata filters by having the LLM map the question to a predefined
# metadata template with attributes we stored during ingestion. This approach
# offers greater flexibility than the SelfQueryRetriever but requires more setup. First,
# import the libraries necessary to create a structured query with specific filters,

# The DestinationSearch class translates the user question into a structured object with
# a content_search field containing the question (minus filtering details) and fields for
# inferred search filters.

#### Step 1-2-3-2: Conversion of user question to structured query including metadata filter
# BUILDING A QUERY CHAIN TO CONVERT THE QUESTION INTO A STRUCTURED QUERY
# define the query generator chain to convert the user question into a structured query with metadata filters.

system_message = """You are an expert at converting user 
questions into vector database queries. 
You have access to a database of tourism destinations.
Given a question, return a database query optimized 
to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, 
do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{question}"),
    ]
)
structured_llm = llm.with_structured_output(
    DestinationSearch, method="function_calling")
query_generator = prompt | structured_llm

# try out the chain with the same question used earlier
question = "Tell me about events or festivals in the UK town of Newquay"
structured_query =query_generator.invoke(question)
structured_query

# With the structured query created, generate a ChromaDB-compatible search filter
search_filter = build_filter(structured_query)
search_filter

# Perform the vector search using the generated structured query and ChromaDB filter
search_query = structured_query.content_search
search_query

metadata_retriever = uk_with_metadata_collection.as_retriever(
    search_kwargs={'k':3, 'filter': search_filter})
answer = metadata_retriever.invoke(search_query)
print(answer)
## COMMENT: this is only the retrieval step; we still need to wrap it in a RAG chain

#endregion main 2: Self metadata query

#region main 2: Generating a structured SQL query
## Step 2-1: Connecting to the UkBooking database
# from langchain_community.utilities import SQLDatabase
# from langchain_classic.chains import create_sql_query_chain
# from langchain_community.tools import QuerySQLDataBaseTool

# Many LLMs can transform user questions into SQL queries, enabling access to relational
# databases directly from LLM applications. While LLMs are continually improving
# in generating accurate SQL, challenges remain, especially when working with
# complex schemas or specific database structures. LangChain enhances these
# capabilities with evolving text-to-SQL features, but there are some common issues we should consider.


# Installing SQLite:
# SQLite doesn’t require full installation. Unzip the package, place it in a folder, and
# add the folder to our system’s Path environment variable.

# Open our operating system shell, navigate to the code folder, and enter the following
# command to create the UkBooking database
# This opens the SQLite terminal: sqlite3 UkBooking.db


# In the SQLite terminal, load the SQL scripts to create and populate the UkBooking database
# To confirm the setup, check for records in the Offer table:
# sqlite> SELECT * FROM Offer;
# Now the UkBooking database is ready for use with LangChain.

db = SQLDatabase.from_uri("sqlite:///UkBooking.db")
print(db.get_usable_table_names())

db.run("SELECT * FROM Offer;")

# Using the CREATE TABLE command along with sample data helps the LLM better understand
# the structure and constraints, minimizing incorrect column and table references

## Step 2-2: Generate SQL queries from natural language
### Step 2-2-1: Generating the SQL query
# generating SQL queries directly from natural language questions.

sql_query_gen_chain = create_sql_query_chain(llm, db)
response = sql_query_gen_chain.invoke(
    {"question": 
     "Give me some offers for Cardiff, including the hotel name"})
response

# if we attempt to execute this SQL directly against the database, you’ll
# encounter an error due to the backticks (```), which are non-SQL characters
#db.run(response) # returns error

### Step 2-2-2: Executing the SQL query [NOTE: THIS WILL THROW AN ERROR]
sql_query_exec_chain = QuerySQLDataBaseTool(db=db)
sql_query_gen_chain = create_sql_query_chain(llm, db)
chain = sql_query_gen_chain | sql_query_exec_chain
chain.invoke({"question": "Give me some offers for Cardiff, including the hotel name"})

### Step 2-2-3: Fixing the SQL format
# To clean up the SQL formatting, we can use the LLM to strip unnecessary characters
# and output a properly formatted SQL statement.

clean_sql_prompt_template = """You are an expert in SQL Lite. 
You are asked to fix badly formed SQL Lite queries, 
which might contain unneded prefixes or suffixes. 
Given the following unclean SQL statement, 
transform it to a clean, 
executable SQL statement for SQL lite.
Always prefix column names with the table name.
Only return an executable SQL statement which terminates 
with a semicolon. Do not return anything else.
Do not include the language name or symbols like ```.

Unclean SQL: {unclean_sql}"""

clean_sql_prompt = ChatPromptTemplate.from_template(
    clean_sql_prompt_template)

clean_sql_chain = clean_sql_prompt | llm

full_sql_gen_chain = sql_query_gen_chain | \
   clean_sql_chain | StrOutputParser()

# try out this full chain with a sample question and verify the output:
question = """Give me some offers for Cardiff, 
including the accomodation name"""
response = full_sql_gen_chain.invoke({"question": question})
print(response)
### Comment: now SQL is fixed

# This approach ensures that the SQL statement is correctly formatted and ready to execute
# against the database.

### Step 2-2-4: Executing the SQL query
# create a chain to generate and execute SQL queries.
sql_query_exec_chain = QuerySQLDataBaseTool(db=db)

sql_query_gen_and_exec_chain = full_sql_gen_chain \
    | sql_query_exec_chain | StrOutputParser()

response = sql_query_gen_and_exec_chain.invoke(
    {"question":question})
response
## COMMENT: this is only the retrieval step; we still need to wrap it in a RAG chain

# This setup allows we to retrieve data from a relational database by using a combined
# chain (sql_query_gen_and_exec_chain) that handles both SQL generation and execution.
# we can easily integrate this chain within a broader RAG setup, as discussed in
# earlier sections. The sequence diagram in figure 10.3 gives us a visual idea of what
# the full RAG with SQL workflow would look like. Try extending this integration as an
# exercise.

# TIP: LangChain’s SQLDatabaseChain class provides a streamlined way to generate
# SQL queries directly from user questions. This tool uses an LLM and
# our database connection to automatically create few-shot prompts, similar to
# those recommended in the Rajkumar paper. Experimenting with SQLDatabaseChain
# can be highly beneficial if we plan to incorporate relational
# databases into our RAG setup.

#endregion main 2: Generating a structured SQL query

#region main 3: Query router
# In the previous section, we learned how to generate SQL queries from natural language.
# However, these queries rely on strict SQL, meaning they depend on exact
# matching and traditional relational operations. Relational databases operate on
# record sets using operations such as SELECT, JOIN, WHERE, and GROUP BY, where filters
# are based on exact string matches or numeric comparisons.

# But what if we want to expand the SQL search to include results that are similar in
# meaning to what the user intended? This requires a shift from standard SQL to a
# semantic SQL search.


# Semantic SQL query (semantic SQL search or SQL similarity search):
# With the rise of LLMs, several relational databases now support semantic search,
# which enables searches based on embeddings instead of exact matches. An example is
# pgvector, an extension for PostgreSQL that allows vector-based similarity searches using
# metrics such as Euclidean or cosine distance. This approach enables us to perform
# searches that return results based on meaning rather than exact text matches.


# Creating the embeddings:
# Using LangChain’s OpenAIEmbeddings wrapper
YOUR_DB_CONNECTION_STRING = ""
db = SQLDatabase.from_uri(YOUR_DB_CONNECTION_STRING)
embeddings_model = OpenAIEmbeddings() # Instantiates the database client and embeddings model

first_names_resultset_str = db.run('SELECT first_name FROM user')
first_names = [fn[0] for fn in eval(first_names_resultset_str)] # Extracts a list of strings from the SQL result string

first_names_embeddings = embeddings_model.embed_documents(first_names) # Calculates the embedding of each first name

fn_emb = zip(first_names,first_names_embeddings) # Associates the first names with the related embeddings

for fn, emb in fn_emb:
    sql = f'UPDATE user SET first_name_embeddings = ARRAY{emb} WHERE first_name ="{fn}"'
    db.run(sql)

# By following these steps, you’ll enable semantic search on first_name or other fields,
# allowing pgvector to retrieve records based on similarity, rather than exact matches.
    
# Performing a semantic SQL search:
# After setting up the embeddings (and indexing the related column to guarantee adequate
# performance on big datasets), we can perform a similarity search as follows:
    
embedded_query= embeddings_model.embed_query("Roberto")
query = ('SELECT first_name FROM user WHERE first_name_embeddings IS NOT NULL ORDER BY first_name_embeddings <-> "{embedded_query}"'
)
db.run(query)

# Automating semantic SQL search:
# After understand how to generate embeddings and perform similarity
# searches in a SQL database, the final step is to create a prompt that can automatically
# generate SQL similarity queries. This process is similar to what we covered for generating
# traditional SQL queries. Once we design, implement, and test this prompt—
# and integrate it into a full chain within LangChain Expression Language (LCEL)—
# our LLM application will be capable of generating semantic searches on pgvector or
# any SQL database that supports ARRAY (or similar) data types, seamlessly feeding the
# results to the LLM for synthesis.

# Benefits of a semantic SQL search:
# The simple example here only scratches the surface of semantic SQL’s capabilities.
# we can combine semantic filtering with exact matching or use multiple semantic filters,
# which are especially powerful in multi-table queries using joins. This approach
# allows highly nuanced searches, especially when combined with traditional SQL
# filtering.

# Combine metadata and semantic filtering in a vector
# store, which can achieve similar results. However, using multiple semantic filters in
# SQL offers greater flexibility, particularly for complex queries.

# Generating queries for a graph database:

## Step 3-1: Setting up the data retrievers (For Chain routing)
### Step 3-1-1: Setting up the vector store retriever
tourism_info_retriever_chain = RunnableLambda(
    lambda x: x['question']) \
       | uk_with_metadata_collection.as_retriever(
           search_kwargs={'k':2}) 

### Step 3-1-2: Setting up the relational database retriever (Same as sql_query_gen_and_exec_chain above)
uk_accommodation_retriever_chain =  full_sql_gen_chain \
    | sql_query_exec_chain | StrOutputParser()

## Step 3-2: Setting up the query router (For Chain routing)

structured_llm_router = llm.with_structured_output(
    RouteQuery) #A
#A Structured router which uses LLM function calls

### Step 3-2-1: Setting up the question router chain
system = """You are an expert at routing a user question 
to a tourism info vector store 
or to an UK accommodation booking relational database.
The vector store contains tourism information about UK destinations.
Use the vectorstore for general tourism information questions 
on UK destinations. 
For questions about accommodation availability or booking, 
use the UK Booking database."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

### Step 3-2-2: Testing the router chain
selected_data_source = question_router.invoke(
    {"question": "Have you got any offers in Brighton?"}
)
print(selected_data_source)
selected_data_source = question_router.invoke(
    {"question": "Where are the best beaches in Cornwall?"}
)
print(selected_data_source)

### Step 3-2-3: Setting up the retriever chooser
retriever_chains = {
    'tourism_info_store': tourism_info_retriever_chain,
    'uk_booking_db': uk_accommodation_retriever_chain
}

chosen = retriever_chooser("""Tell me about events 
or festivals in the UK town of Newquay""") 
print(chosen)


## Step 3-3: Setting up the full RAG chain and Integrating the chain router (For Chain routing)
# from langchain_core.runnables import RunnablePassthrough

rag_prompt_template = """
Given a question and some context, answer the question.
If you get a structured context, like a tuple, try to 
infer the meaning of the components: 
typically they refer to accommodation offers, 
and the number is a percentage (0.2 means 20%).
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

## Step 3-4: Executing the full RAG chain 
### Step 3-4-1: Question on accommodation offers
question = """Give me some offers for Cardiff, 
including the accommodation name"""

chosen_retriever = retriever_chooser(question)

answer = execute_rag_chain(question, chosen_retriever)
print(answer)

### Step 3-4-2: Question on tourism information
question_2 = """Tell me about events or festivals 
in the UK town of Newquay"""

chosen_retriever_2 = retriever_chooser(question_2)

answer2 = execute_rag_chain(question_2, chosen_retriever_2)
print(answer2)

#endregion main 3:Query router

#region main 3:Retrieval post processing
## Step 4-1: RAG Fusion
### Step 4-1-1: Multiple query Generation (same as for MultiQueryRetriver)

multi_query_gen_prompt_template = """
You are an AI language model assistant. Your task is 
to generate five different versions of the given user 
question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the 
user question, your goal is to help
the user overcome some of the limitations of the 
distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}
"""

multi_query_gen_prompt = ChatPromptTemplate.from_template(multi_query_gen_prompt_template) 
questions_parser = LineListOutputParser()
multi_query_gen_chain = multi_query_gen_prompt | llm | questions_parser

# With this setup, we can now generate multiple alternative queries from a single question,
# helping to capture varied perspectives and nuances that improve document
# retrieval accuracy.
# NOTE I’ve chosen to use GPT-5 instead of GPT-5-mini or GPT-5-nano, as it’s
# more likely to produce higher-quality queries and generate more accurate,
# well-synthesized responses


# Now that multiple queries can be generated, the next step is to implement a ranking
# mechanism to sort and prioritize the retrieved results. We’ll use the RRF algorithm for
# ranking.

### Step 4-1-2: Reciprocal Rank Fusion (RRF) algorithm
# RRF ALGORITHM: => def reciprocal_rank_fusion(results_groups: list[list], k=60):
# The core of this workflow is the RRF algorithm, which assigns scores to documents
# retrieved by multiple queries. Using the RRF formula, each document is scored based
# on its rank and then reranked by total RRF score. See the following listing for implementation
# details.

# Based on: https://github.com/Raudaschl/rag-fusion/blob/master/main.py


# SETING UP THE RAG FUSION RETRIEVAL CHAIN:
# With the RRF algorithm in place, let’s create a RAG fusion retrieval chain
retriever = uk_with_metadata_collection.as_retriever(
    search_kwargs={'k':3})
top_three_results = RunnableLambda(
    lambda x: x[0:3]) # select the top three results

rag_fusion_retrieval_chain = multi_query_gen_chain \
    | retriever.map() | reciprocal_rank_fusion \
    | top_three_results # Full RAG fusion retrieval chain
        
docs = rag_fusion_retrieval_chain.invoke(
    {"question": question}) # testing the retrieval_chain_rag_fusion chain
len(docs)

### Step 4-1-3: Incorporating/Integrating Rag Fusion into the RAG Chain
# The final step is to integrate this RAG Fusion retrieval chain into a larger RAG chain
# for end-to-end question routing, retrieval, and answer synthesis.

# integrating a retrieval chain into a broader RAG chain is
# straightforward. For completeness, the following listing shows how to incorporate the
# RAG fusion retrieval chain within a RAG chain.


    
rag_prompt_template = """
Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

rag_chain = (
    {
        "context": {"question": RunnablePassthrough()} | rag_fusion_retrieval_chain,# The context is returned by the retriver after feeding to it the step-back question
        "question": RunnablePassthrough(),# This is the original user question
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# test the complete RAG chain with an example question
user_question = "Can you give me some tips for a trip to Brighton?"

answer = rag_chain.invoke(user_question)
print(answer)


#endregion main 3:Retrieval post processing