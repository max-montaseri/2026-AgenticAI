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

#region main
openai_api_key = get_env_api_key ("OpenAI")
llm_model = get_llm_model("GPT-Cheapest")
llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)

## Splitting (Document hierarchy strategy) and ingesting the content of a single URL (on Cornwall)
# Splitting strategy => Document hierarchy
# Splitting by HTML header

# create a ChromaDB collection to store the more granular chunks
corwnall_granular_chormaDb_collection = create_chormaDb_collection_and_reset("cornwall_granular", openai_api_key)

# set up a second collection for coarser chunks
corwnall_coarse_chormaDb_collection = create_chormaDb_collection_and_reset("cornwall_coarse", openai_api_key)

coarse_html_url_address = "https://en.wikivoyage.org/wiki/Cornwall"
coarse_html_docs = load_html_content_with_asyncHtmlLoader(coarse_html_url_address)

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]

# plitting content with the HTMLSectionSplitter
html_section_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

# generate the granular chunks:
coarse_granular_chunks = split_html_docs_into_granular_chunks(coarse_html_docs, html_section_splitter)

# insert the granular chunks into the Chroma collection:
corwnall_granular_chormaDb_collection.add_documents(documents=coarse_granular_chunks)

# SEARCHING GRANULAR CHUNKS
similarity_search_results = corwnall_granular_chormaDb_collection.similarity_search(
    query="Events or festivals in Cornwall",k=3)

for doc in similarity_search_results:
    print(doc)

# Splitting by HTML header - Step 5: Splitting into coarse chunks with the RecursiveCharacterTextSplitter 
# For larger, coarser chunks, use the RecursiveCharacterTextSplitter class. 

# Start by creating the necessary objects:
html2text_transformer = Html2TextTransformer()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=300
)
# generate the coarse chunks
coarse_chunks = split_html_docs_into_chunks(coarse_html_docs, html2text_transformer, text_splitter)

# insert the granular chunks into the Chroma collection:
corwnall_granular_chormaDb_collection.add_documents(documents=coarse_granular_chunks)

# SEARCHING GRANULAR CHUNKS
granular_chunks_similarity_search_results = corwnall_granular_chormaDb_collection.similarity_search(
    query="Events or festivals in Cornwall",k=3)
for doc in granular_chunks_similarity_search_results:
    print(doc)



## Splitting (Document hierarchy strategy) and ingesting the content of various 
# URLs (across UK destinations)
    
#  Creating collections for multiple UK destinations

# how to set up new granular and coarse collections for various
# UK destinations and ingest the related content chunks. If we'd like to minimize processing
# costs, consider reducing the size of the uk_destinations list.


uk_granular_chormaDb_collection = create_chormaDb_collection_and_reset("uk_granular", openai_api_key)
uk_coarse_chormaDb_collection = create_chormaDb_collection_and_reset("uk_coarse", openai_api_key)

# Splitting and ingesting HTML content with the HTMLSectionSplitter
# Reduce this list if we want to save on processing fees
uk_destinations = [
    "Cornwall", "North_Cornwall", "South_Cornwall", "West_Cornwall", 
    "Tintagel", "Bodmin", "Wadebridge", "Penzance", "Newquay",
    "St_Ives", "Port_Isaac", "Looe", "Polperro", "Porthleven"
    "East_Sussex", "Brighton", "Battle", "Hastings_(England)", 
    "Rye_(England)", "Seaford", "Ashdown_Forest"
]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(destination_url) # Loader for one destination
    docs =  html_loader.load() # Documents of one destination 
    
    for doc in docs:
        print(doc.metadata)
        granular_chunks = split_html_docs_into_granular_chunks(docs, text_splitter)
        uk_granular_chormaDb_collection.add_documents(documents=granular_chunks)
        coarse_chunks = split_html_docs_into_chunks(docs)
        uk_coarse_chormaDb_collection.add_documents(documents=coarse_chunks)

# Searching
# perform both granular and coarse searches
granular_results = uk_granular_chormaDb_collection.similarity_search(
    query="Events or festivals in East Sussex",k=4)
for doc in granular_results:
    print(doc)

coarse_results = uk_coarse_chormaDb_collection.similarity_search(
    query="Events or festivals in East Sussex",k=4)
for doc in coarse_results:
    print(doc)


#endregion