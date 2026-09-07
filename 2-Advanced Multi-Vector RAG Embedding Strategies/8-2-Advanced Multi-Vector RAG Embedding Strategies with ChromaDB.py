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
def split_html_docs_into_chunks(docs, html2text_transformer, text_splitter):
    # define a function to split the content into coarse chunks
    text_docs = html2text_transformer.transform_documents(docs) # transform HTML docs into clean text docs 
    chunks = text_splitter.split_documents(text_docs)

    return chunks

def split_html_docs_into_granular_chunks(docs, text_splitter):
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content # Extract the HTML text from the document
        temp_chunks = text_splitter.split_text(
            html_string) # Each chunk is a H1 or H2 HTML section
        all_chunks.extend(temp_chunks) 

    return all_chunks

def load_html_content_with_asyncHtmlLoader(urlAddress):
    # from langchain_community.document_loaders import AsyncHtmlLoader

    # Loading the HTML content with the AsyncHtmlLoader
    # The final step is to ingest some content about Cornwall, a region in the UK known for
    # its stunning seaside resorts, using an HTML loader:
    destination_url = urlAddress
    # html_loader = AsyncHtmlLoader(destination_url)

    # Disable SSL verification (development only):
    # If we just experimenting, we can disable SSL verification. Do not do this in production.
    html_loader = AsyncHtmlLoader(
        destination_url,
        verify_ssl=False
    )

    # response = requests.get(destination_url)
    # print(response.status_code)

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
    
#region Chains
#endregion   

openai_api_key = get_env_api_key ("OpenAI")
llm_model = get_llm_model("GPT-Cheapest")
llm = get_llm()

uk_destinations = [
    "Cornwall", "North_Cornwall", "South_Cornwall", "West_Cornwall", 
    "Tintagel", "Bodmin", "Wadebridge", "Penzance", "Newquay",
    "St_Ives", "Port_Isaac", "Looe", "Polperro", "Porthleven"
    "East_Sussex", "Brighton", "Battle", "Hastings_(England)", 
    "Rye_(England)", "Seaford", "Ashdown_Forest"
]
wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

#region main 1-Embedding child chunks with ParentDocumentRetriever

## Embedding child chunks with ParentDocumentRetriever
# A common challenge with chunk size is balancing between context and detail. Large
# chunks work for broad questions but struggle with detailed queries. Small chunks,
# while supporting detailed queries, often lack the context needed for generating
# comprehensive answers. This creates a tradeoff—if chunks are too small, the response
# might be incomplete, but if they’re too large, the retrieval may be less precise.

# To solve this problem, split the document into larger parent chunks, and create
# smaller child chunks within each parent. Use the child chunks solely for generating
# more granular embeddings, which are then stored against the parent chunk. This
# hybrid approach allows each document to have embeddings for both broad and
# detailed queries,

# Step 1: Setting up the Parent Document retriever
# This approach uses two types of stores: a document store, which holds the complete parent
# documents, and a vector store, which contains the smaller chunks and their corresponding
# embeddings. 

# Each chunk maintains a reference to its parent document.
# The approach begins by splitting content into large, coarse chunks for synthesis and
# then further dividing each into smaller child chunks for retrieval.

# Splitter to generate parent coarse chunks from original documents (parsed from web pages)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)

# Splitter to generate child granular chunks from parent coarse chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500) 

child_chunks_collection = create_chormaDb_collection_and_reset("uk_child_chunks", openai_api_key)

# Document store to host parent coarse chunks
doc_store = InMemoryStore() 

# Retriever to link parent coarse chunks to child granular chunks
parent_doc_retriever = ParentDocumentRetriever( 
    vectorstore=child_chunks_collection,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Ingesting the content into doc and vector store
# Now generate both coarse and granular chunks using the configured splitters, and
# add them to the respective stores. This happens in each parent_doc_retriever.add_
# documents() call, for the corresponding destination URL
html2text_transformer = Html2TextTransformer()

for destination_url in uk_destination_urls:
    html_docs = load_html_content_with_asyncHtmlLoader(destination_url)
    text_docs = html2text_transformer.transform_documents(html_docs) # Transform HTML docs into clean text deocs

    print(f'Ingesting {destination_url}')
    parent_doc_retriever.add_documents(text_docs, ids=None) # Ingest coarse chunks into document store and granular chunks into vector store

# VERIFYING THE IN-MEMORY DOCUMENT STORE
# check if the coarse chunks have been correctly stored
list(doc_store.yield_keys()) # Show the keys of the added coarse chunks

# Step 4: Performing a search on granular information
# perform a search on the child chunks using the ParentDocumentRetriever:
retrieved_docs = parent_doc_retriever.invoke("Cornwall Ranger")
len(retrieved_docs)
retrieved_docs[0]


# Step 5: Comparing with direct semantic search on child chunks
# compare the results by directly searching only the child chunks:
child_docs_only =  child_chunks_collection.similarity_search("Cornwall Ranger")
len(child_docs_only)
child_docs_only[0]

# The result we’ve obtained with the ParentDocumentRetriever is particularly useful
# when used as context for LLM synthesis, as it provides broader details around the specific
# information.

#endregion main 1-Embedding child chunks with ParentDocumentRetriever

#region main 2-Embedding child chunks with MultiVectorRetriever

# An alternative method for embedding child chunks and linking them to the larger
# parent chunks used in synthesis is to use the MultiVectorRetriever. Begin by importing
# the necessary libraries, including InMemoryByteStore, which is specifically
# designed for storing binary data. In this store, keys are strings and values are bytes,
# making it ideal for use cases involving embeddings, models, or files where raw byte
# storage is preferred or required

# from langchain_classic.storage import InMemoryByteStore
# from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
# import uuid

doc_byte_store = InMemoryByteStore() # Document store to host parent coarse chunks
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( # Retriever to link parent coarse chunks to child granular chunks
    vectorstore=child_chunks_collection,
    byte_store=doc_byte_store
)

# Ingesting the content into doc and vector store
# While ingestion may feel slower, this is expected due to the added complexity
# of managing multiple vector representations.

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) # Loader for one destination
    html_docs =  html_loader.load() # Documents of one destination
    text_docs = html2text_transformer.transform_documents(
        html_docs) # transform HTML docs into clean text docs

    coarse_chunks = parent_splitter.split_documents(
        text_docs) # Split the destination content into parent coarse chunks

    coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
    all_granular_chunks = []
    for i, coarse_chunk in enumerate(
        coarse_chunks): # Iterate over the parent coarse chunks
        
        coarse_chunk_id = coarse_chunks_ids[i]
            
        granular_chunks = child_splitter.split_documents(
            [coarse_chunk]) # Create child granular chunks form each parent coarse chunk

        for granular_chunk in granular_chunks:
            granular_chunk.metadata[doc_key] = coarse_chunk_id # Link each child granular chunk to its parent coarse chunk

        all_granular_chunks.extend(granular_chunks)

    print(f'Ingesting {destination_url}')
    multi_vector_retriever.vectorstore.add_documents(
        all_granular_chunks) # Ingest the child granular chunks into the vector store
    multi_vector_retriever.docstore.mset(
        list(zip(coarse_chunks_ids, coarse_chunks))) # Ingest the parent coarse chunks into the document store

# Performing a search on granular information
# Now perform a search using MultiVectorRetriever, just like we did with the Parent-
# DocumentRetriever

retrieved_docs = multi_vector_retriever.invoke(
    "Cornwall Ranger")
len(retrieved_docs)
retrieved_docs[0]

# Step 4: Comparing with direct semantic search on child chunks
child_docs_only =  child_chunks_collection.similarity_search(
    "Cornwall Ranger")
len(child_docs_only)
child_docs_only[0]

#endregion main 2-Embedding child chunks with MultiVectorRetriever

#region main 3-Embedding summaries with MultiVectorRetriever

# from langchain_core.documents import Document

# Embeddings from coarse chunks are often ineffective because they capture too much
# irrelevant content. A large chunk may include filler text or minor details that dilute
# the semantic value of the embeddings, making them less focused and less useful.
# To address this, we can create a summary of the coarse chunk and generate
# embeddings from it. These summary embeddings are then stored alongside the original
# chunk embeddings.

# Because the summary is more concise and relevant, the resulting embeddings are denser and more effective for retrieval,
# reducing noise and improving search precision.

# Vector store collection to host child granular chunks
summaries_collection = create_chormaDb_collection_and_reset("uk_summaries", openai_api_key)

doc_byte_store = InMemoryByteStore() # Document store to host parent coarse chunks
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( # Retriever to link parent coarse chunks to child granular chunks
    vectorstore=summaries_collection,
    byte_store=doc_byte_store
)

# Setting up the summarization chain
# Use an LLM to generate summaries of the coarse chunks. Define a summarization
# chain that extracts the content, prompts the LLM, and parses the response into a
# usable format

summarization_chain = (
    {"document": lambda x: x.page_content} # Grab the text content from the document
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{document}") # Instantiate a prompt asking to generate summary of the provided text
    | llm # Send the LLM the instantiated prompt
    | StrOutputParser()) # Extract the summary text from the response

# Ingesting the coarse chunks and related summaries into doc and vector stores

# load the content, split it into coarse chunks, and generate summaries for those
# chunks. Then, store the summaries in the document store while storing the corresponding
# coarse chunks in the vector store

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) # Loader for one destination
    html_docs =  html_loader.load() # Documents of one destination
    text_docs = html2text_transformer.transform_documents(
        html_docs) # transform HTML docs into clean text docs

    coarse_chunks = parent_splitter.split_documents(
        text_docs) # Split the destination content into coarse chunks

    coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
    all_summaries = []
    for i, coarse_chunk in enumerate(
        coarse_chunks): # Iterate over the coarse chunks
        
        coarse_chunk_id = coarse_chunks_ids[i]
            
        summary_text =  summarization_chain.invoke(
            coarse_chunk) # Generate a summary for the coarse chunk thorugh the summarization chain
        summary_doc = Document(page_content=summary_text, 
                               metadata={doc_key: coarse_chunk_id})

        all_summaries.append(summary_doc) # Link each summary to its related coarse chunk

    print(f'Ingesting {destination_url}')
    multi_vector_retriever.vectorstore.add_documents(
        all_summaries) # Ingest the summaries into the vector store
    multi_vector_retriever.docstore.mset(
        list(zip(coarse_chunks_ids, coarse_chunks))) # Ingest the coarse chunks into the document store

# COMMENT: the code above is similar to when ingesting child chunks, but it is slower because of the summarization step
# which invokes the LLM.
# The processing can be speeded up by parallelizing the outer for loop on the destination urls.


# Performing a search on granular information
# Once the ingestion is complete, we can perform a search using the MultiVector-
# Retriever, which now uses the summaries for each travel destination

retrieved_docs = multi_vector_retriever.invoke("Cornwall travel")
len(retrieved_docs)


# Comparing with direct semantic search on summaries

# If we print the first result (retrieved_docs_only[0]), we’ll see a large chunk similar
# to those retrieved when using child embeddings. These larger chunks provide more
# context, making them effective when passed as input to the LLM.

# For comparison, perform a direct search on the summaries alone:
summary_docs_only =  summaries_collection.similarity_search(
    "Cornwall Travel")
len(summary_docs_only)
print(summary_docs_only[0])

#endregion main 3-Embedding summaries with MultiVectorRetriever

#region main 4-Embedding hypothetical questions with MultiVectorRetriever

# from pydantic import BaseModel, Field

# When querying a vector store, our natural language question is converted into a vector,
# and the system calculates its similarity (e.g., cosine distance) to the stored vectors.
# The documents linked to the closest vectors are then returned. This approach works
# well if the question is semantically similar to the ideal answer. But often, the wording
# of the question and the phrasing of the ideal answer may not match closely enough,
# causing the search to miss relevant documents.

# To address this, we can generate hypothetical questions that each chunk is likely
# to answer and then store the chunk using embeddings derived from these questions. 
# This method increases the chances that the stored vectors will
# align more closely with a user’s query, making it more likely to retrieve relevant information,
# even if the original document’s embeddings aren’t a perfect match for the
# question.

# Step 1: Setting up the Multi vector retriever (same as when embedding summaries)
# Set up the MultiVectorRetriever similarly to how it was configured for summary
# embeddings, but this time, use a vector store specifically for storing hypothetical questions,
# as shown in the following listing.

# MultiVectorRetriever with hypothetical questions

# Vector store collection to host child granular chunks
hypothetical_questions_collection = create_chormaDb_collection_and_reset("uk_summaries", openai_api_key)

doc_byte_store = InMemoryByteStore() # Document store to host parent coarse chunks
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( # Retriever to link parent coarse chunks to child granular chunks
    vectorstore=hypothetical_questions_collection,
    byte_store=doc_byte_store
)

# Setting up the chain to generate hypothetical questions

# Create a chain to generate hypothetical questions for each document chunk. Use
# structured output from the LLM to ensure the generated questions are returned

class HypotheticalQuestions(BaseModel):
    """A list of hypotetical questions for given text."""

    questions: List[str] = Field(..., description="List of hypothetical questions for given text")

llm_with_structured_output = get_llm_with_structured_output(HypotheticalQuestions)

hypothetical_questions_chain = (
    {"document_text": lambda x: x.page_content} # Grab the text content from the document
    | ChatPromptTemplate.from_template( # Instantiate a prompt asking to generate 4 hypothetical questions on the provided text
        "Generate a list of exactly 4 hypothetical questions that the below text could be used to answer:\n\n{document_text}"
    )
    | llm_with_structured_output # Invoke the LLM configured to return an object containing the questions as a typed list of strings
    | (lambda x: x.questions) # Grab the list of questions from the response
)

# Ingesting the coarse chunks and related hypothetical questions into doc and vector store
# Now generate coarse chunks, create the hypothetical questions for each, and store
# them in the respective collections, as shown in the following listing.

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) # Loader for one destination
    html_docs =  html_loader.load() # Documents of one destination 
    text_docs = html2text_transformer.transform_documents(
        html_docs) # transform HTML docs into clean text docs

    coarse_chunks = parent_splitter.split_documents(
        text_docs) # Split the destination content into coarse chunks

    coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
    all_hypothetical_questions = []
    for i, coarse_chunk in enumerate(
        coarse_chunks): # Iterate over the coarse chunks
        
        coarse_chunk_id = coarse_chunks_ids[i]
            
        hypothetical_questions = hypothetical_questions_chain.invoke(
            coarse_chunk) # Generate a list of hypothetical questions for the coarse chunk thorugh the question generation chain
        hypothetical_questions_docs = [Document(
            page_content=question, metadata={doc_key: coarse_chunk_id})
                    for question 
                    in hypothetical_questions] # Link each hypothetical question to its related coarse chunk

        all_hypothetical_questions.extend(hypothetical_questions_docs)

    print(f'Ingesting {destination_url}')
    multi_vector_retriever.vectorstore.add_documents(
        all_hypothetical_questions) # Ingest the hypothetical questions into the vector store
    multi_vector_retriever.docstore.mset(
        list(zip(coarse_chunks_ids, coarse_chunks))) # Ingest the coarse chunks into the document store

# Performing a search on granular information
# After ingestion, perform a search using the MultiVectorRetriever, which now uses the
# stored hypothetical question embeddings:

retrieved_docs = multi_vector_retriever.invoke(
    "How can you go to Brighton from London?")
len(retrieved_docs)

# Step 5: Inspecting possible questions matching our question through semantic search

# COMPARING WITH DIRECT SEARCH ON HYPOTHETICAL QUESTIONS
# run a search directly on the hypothetical questions collection for comparison:
hypothetical_question_docs_only = hypothetical_questions_collection.similarity_search(
"How can you go to Brighton from London?")
len(hypothetical_question_docs_only)
# The results show hypothetical questions closely matching the query:

#endregion main 4-Embedding hypothetical questions with MultiVectorRetriever

# region main 5-Granular chunk expansion with MultiVectorRetriever

# the main drawback of splitting a document into small granular
# chunks is that while these chunks are effective for detailed questions, they often
# lack the context needed to generate complete answers. One way to address this is
# through chunk expansion

# The idea is to store an expanded version of each chunk that includes content from
# the chunks immediately before and after it. This expanded version is stored in a separate
# document store. So when the vector store retrieves a relevant granular chunk, the
# linked expanded chunk is returned instead, offering a richer context for the LLM to
# produce a more complete answer.

# This technique can be easily implemented using the MultiVectorRetriever.

# Step 1: Setting up the Multi vector retriever

# configure the MultiVectorRetriever by creating a collection to hold granular
# chunks and an in-memory document store for the expanded chunks.
granular_chunk_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500) # Splitter to generate granular chunks from original documents (parsed from web pages)

# Vector store collection to host child granular chunks
granular_chunks_collection = create_chormaDb_collection_and_reset("uk_granular_chunks", openai_api_key)

expanded_chunk_store = InMemoryByteStore() # Document store to host expanded chunks
doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( # Retriever to link parent coarse chunks to child granular chunks
    vectorstore=granular_chunks_collection,
    byte_store=expanded_chunk_store
)

# Step 2: Ingesting granular and expanded chunks into doc and vector store
for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) # Loader for one destination
    html_docs =  html_loader.load() # Documents of one destination
    text_docs = html2text_transformer.transform_documents(
        html_docs) # transform HTML docs into clean text docs

    granular_chunks = granular_chunk_splitter.split_documents(
        text_docs) # Split the destination content into granular chunks

    expanded_chunk_store_items = []
    for i, granular_chunk in enumerate(
        granular_chunks): # Iterate over the granular chunks

        this_chunk_num = i # determine the index of the current chunk and its previous and next chunks
        previous_chunk_num = i-1 # determine the index of the current chunk and its previous and next chunks
        next_chunk_num = i+1 # determine the index of the current chunk and its previous and next chunks
        
        if i==0: #F
            previous_chunk_num = None
        elif i==(len(granular_chunks)-1): # determine the index of the current chunk and its previous and next chunks
            next_chunk_num = None

        expanded_chunk_text = "" # Assemble the text of the expanded chunk by including the previous and next chunk
        if previous_chunk_num: # Assemble the text of the expanded chunk by including the previous and next chunk
            expanded_chunk_text += granular_chunks[
                previous_chunk_num].page_content
            expanded_chunk_text += "\n"

        expanded_chunk_text += granular_chunks[
            this_chunk_num].page_content # Assemble the text of the expanded chunk by including the previous and next chunk
        expanded_chunk_text += "\n"

        if next_chunk_num: # Assemble the text of the expanded chunk by including the previous and next chunk
            expanded_chunk_text += granular_chunks[
                next_chunk_num].page_content
            expanded_chunk_text += "\n"

        expanded_chunk_id = str(uuid.uuid4()) # Generate the ID of the expanded chunk
        expanded_chunk_doc = Document(
            page_content=expanded_chunk_text) # Create the expanded chunk document

        expanded_chunk_store_item = (expanded_chunk_id, 
                                     expanded_chunk_doc)
        expanded_chunk_store_items.append(
            expanded_chunk_store_item)

        granular_chunk.metadata[
            doc_key] = expanded_chunk_id # Link each granular chunk to its related expanded chunk
            
    print(f'Ingesting {destination_url}')
    multi_vector_retriever.vectorstore.add_documents(
        granular_chunks) # Ingest the granular chunks into the vector store
    multi_vector_retriever.docstore.mset(
        expanded_chunk_store_items) # Ingest the expanded chunks into the document store

# Step 3: Performing a search on granular information
# After the ingestion step, run a search using the MultiVectorRetriever, which now uses
# expanded chunks for a more complete context:
retrieved_docs = multi_vector_retriever.invoke("Cornwall Ranger")
len(retrieved_docs)
retrieved_docs[0]

# Step 4: Comparing with direct semantic search on granular chunks
# For comparison, run a search directly on the granular chunks without expansion:
child_docs_only =  child_chunks_collection.similarity_search("Cornwall Ranger")
len(child_docs_only)
child_docs_only[0]

# COMMENT: the expanded chunk has more useful context
#endregion main 5-Granular chunk expansion with MultiVectorRetriever