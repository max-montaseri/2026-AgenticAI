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
def split_html_docs_into_granular_chunks(docs, htmlSection):
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content # Extract the HTML text from the document
        temp_chunks = html_section_splitter.split_text(
            html_string) # Each chunk is a H1 or H2 HTML section
        h2_temp_chunks = [chunk for chunk in 
                          temp_chunks if htmlSection
                          in chunk.metadata] # Only keep content associated with H2 sections
        all_chunks.extend(h2_temp_chunks) 

def split_html_docs_into_granular_chunks(docs):
    all_chunks = []
    for doc in docs:
        html_string = doc.page_content # Extract the HTML text from the document
        temp_chunks = html_section_splitter.split_text(
            html_string) # Each chunk is a H1 or H2 HTML section
        all_chunks.extend(temp_chunks) 

    return all_chunks

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

#region main 1-Splitting and ingesting the content of various URLs (across UK destinations)
uk_destinations = [
    "Cornwall", "North_Cornwall", "South_Cornwall", "West_Cornwall", 
    "Tintagel", "Bodmin", "Wadebridge", "Penzance", "Newquay",
    "St_Ives", "Port_Isaac", "Looe", "Polperro", "Porthleven",
    "East_Sussex", "Brighton", "Battle", "Hastings_(England)", 
    "Rye_(England)", "Seaford", "Ashdown_Forest"
]

uk_granular_chormaDb_collection = create_chormaDb_collection_and_reset("uk_granular", openai_api_key)
wikivoyage_root_url = "https://en.wikivoyage.org/wiki"
uk_destination_urls = [f'{wikivoyage_root_url}/{d}' for d in uk_destinations]

headers_to_split_on = [("h1", "Header 1"),("h2", "Header 2")]
html_section_splitter = HTMLSectionSplitter(
    headers_to_split_on=headers_to_split_on)

for destination_url in uk_destination_urls:
    html_loader = AsyncHtmlLoader(
        destination_url) # Loader for one destination
    docs =  html_loader.load() # Documents of one destination
    
    for doc in docs:
        print(doc.metadata)
        granular_chunks = split_html_docs_into_granular_chunks_from_htmlSection(docs, "Header 2")
        uk_granular_chormaDb_collection.add_documents(
            documents=granular_chunks)

#endregion main 1-Splitting and ingesting the content of various URLs (across UK destinations)

#region main 2-Rewrite-retrieve-read

# To improve a poorly worded question, one effective method is to have an LLM rewrite
# it into a clearer form. This approach, covered in “Query Rewriting for Retrieval-
# Augmented Large Language Models” by Xinbei Ma et al., inspired the diagram in figure
# 9.1. In the standard Retrieve-and-Read workflow, the retriever processes the original
# question directly and sends results to the LLM for synthesis. By adding a Rewrite
# step up front, a rewriter (often an LLM) reformulates the question before passing it to
# the retriever. This improved workflow is known as Rewrite-Retrieve-Read.


# Step 1: Retrieving content with original user question
# Let’s start by performing a search with the original user question:
user_question = "Tell me some fun things I can enjoy in Cornwall"
initial_results = uk_granular_chormaDb_collection.similarity_search(
    query=user_question,k=4)
for doc in initial_results:
    print(doc)

# COMMENT: the retrieval from the vector store against the original question is bad
    
# Step 2: Question rewrite
# Step 2-1: Setting up the query rewriter chain
# To refine the user question into a more effective query for ChromaDB, we’ll use the
# LLM to rewrite it into a format better suited for semantic search. Setting up a query
# rewriter chain will help us automate this transformation.
    

rewriter_prompt_template = """
Generate search query for the Chroma DB vector store
from a user question, allowing for a more accurate 
response through semantic search.
Just return the revised Chroma DB query, with quotes around it. 

User question: {user_question}
Revised Chroma DB query:
"""

rewriter_prompt = ChatPromptTemplate.from_template(
    rewriter_prompt_template) 
rewriter_chain = rewriter_prompt | llm | StrOutputParser()

# This setup allows we to pass a user question to the rewriter chain, which generates a
# tailored query optimized for ChromaDB, enhancing retrieval accuracy.

# Step 3: Retrieving content with the rewritten query
# Now let’s use the rewriter chain to create a more targeted query and see if it returns
# more accurate results compared to the original question. First, generate the rewritten
# query:
user_question ="Tell me some fun things I can do in Cornwall"

search_query = rewriter_chain.invoke(
    {"user_question": user_question})
print(search_query)

# Now use this refined query to perform the vector store search:
improved_results = uk_granular_chormaDb_collection.similarity_search(
    query=search_query,k=3)
for doc in improved_results:
    print(doc)

# Step 4: Combining everything in a single RAG chain
# Now we can build a complete workflow that transforms the initial user question into
# a search query for vector retrieval. The original question is retained in the prompt to
# generate the final answer. The following listing shows the full RAG chain, including
# the query rewriting step.

retriever = uk_granular_chormaDb_collection.as_retriever()

rag_prompt_template = """
Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(
    rag_prompt_template) 

rewrite_retrieve_read_rag_chain = (
    {
        "context": {"user_question": RunnablePassthrough()} 
            | rewriter_chain | retriever,# The context is returned by the retriver after feeding to it the rewritten query
        "question": RunnablePassthrough(),# This is the original user question
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

user_question = "Tell me some fun things I can do in Cornwall"

answer = rewrite_retrieve_read_rag_chain.invoke(user_question)
print(answer)

# This output demonstrates a satisfying answer based on the combined Rewrite-
# Retrieve-Read workflow.

#endregion main 2-Rewrite-retrieve-read

#region main 3-Multiple query generation with MultiQueryRetriever

# from langchain_core.output_parsers import BaseOutputParser
# from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# The Rewrite-Retrieve-Read approach assumes the original user question was poorly
# phrased. But if the question is well-formed and contains multiple implicit questions,
# rewriting it as a single improved query may not be effective. In these cases, it’s better
# to have the LLM break down the original question into multiple explicit questions.Each question can be executed separately against the vector store, with the answers
# then synthesized into a comprehensive response. Figure 9.3 illustrates this workflow.

## Step 1: Implementing a custom MultiQueryRetriver (Setting up the chain for generating multiple queries)

# Step 1-1: Setting up the prompt
multi_query_gen_prompt_template = """
You are an AI language model assistant. Your task 
is to generate five different versions of the given 
user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user 
question, your goal is to help the user overcome some of 
the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}
"""

multi_query_gen_prompt = ChatPromptTemplate.from_template(
    multi_query_gen_prompt_template) 

# Step 1-2: Setting up the multi-query parser
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse out a question from each output line."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  

questions_parser = LineListOutputParser()


# Step 1-3: Setting up the chain to generate multiple queries
multi_query_gen_chain = multi_query_gen_prompt | llm | questions_parser


# Step 1-4: Testing the Multi query gen chain
user_question = "Tell me some fun things I can do in Cornwall"
multiple_queries = multi_query_gen_chain.invoke(user_question)

# Step 1-5: Setting up the MultiQueryRetriever
basic_retriever = uk_granular_chormaDb_collection.as_retriever()

multi_query_retriever = MultiQueryRetriever(
    retriever=basic_retriever, llm_chain=multi_query_gen_chain, 
    parser_key="lines" # this is the key for the parsed output
)

# Step 1-6: Using the multi_query retriever
user_question = "Tell me some fun things I can do in Cornwall"
retrieved_docs = multi_query_retriever.invoke(user_question)

##  Step 2: Using directly a standard MultiQueryRetriever 
std_multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=basic_retriever, llm=llm
)
user_question = "Tell me some fun things I can do in Cornwall"

retrieved_docs = std_multi_query_retriever.invoke(user_question)

## Step 3: Step-back question
# When we send a highly detailed question directly to the vector store—assuming our documents are 
# split into small, specific chunks—you might retrieve information that’s too focused and misses the broader context. This can limit the LLM’s ability to generate a comprehensive answer.

# As discussed in section 7.4, one solution is to create two sets of document chunks:
# coarse chunks for synthesis and fine-grained chunks for detailed retrieval. Another solution 
# is to adjust the user question rather than the document chunks, using an approach called a 
# step-back question.

# In this approach, we start with the user’s detailed question but then create a broader question 
# to retrieve a more generalized context. This step-back context provides a higher-level view than 
# the original context derived from the specific question. 

# You then provide the LLM with both the detailed context and the broader context to enable a fuller 
# response, as illustrated in figure 9.4 where the LLM application first sends the 
# detailed question (Q_D) to the vector store to retrieve a detailed context (C_D). 

# It then prompts the LLM to generate a more abstract question (Q_A) based on Q_D, 
# which is also executed in the vector store to obtain an abstract context (C_A). 
# Finally, the LLM application combines Q_D, C_D, and C_A into a single prompt, enabling 
# the LLM to synthesize a comprehensive answer.

# The LLM application sends the original detailed question to the vector store to retrieve 
# detailed context and then generates and executes a broader question to obtain abstract context. 
# It combines both contexts with the original question to enable the LLM to create a comprehensive 
# answer. 

# Step 3: Step-back question
#  Step 3-1: Setting up the chain to generate the step-back question
step_back_prompt_template = """
Generate a less specific question (aka Step-back question) 
for the following detailed question, so that a wider context 
can be retrieved.
Detailed question: {detailed_question}
Step-back question:
"""

step_back_prompt = ChatPromptTemplate.from_template(
    step_back_prompt_template)
step_back_question_gen_chain = step_back_prompt | llm | StrOutputParser()

#  Step 3-2: Testing the step-back-question generation chain
user_question = "Can you give me some tips for a trip to Brighton?"
step_back_question = step_back_question_gen_chain.invoke(user_question)
step_back_question

#  Step 3-3: Incorporating step-back question generation chain into the RAG chain
retriever = uk_granular_chormaDb_collection.as_retriever()

rag_prompt_template = """
Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

step_back_question_rag_chain = (
    {
        "context": {"detailed_question": RunnablePassthrough()} 
           | step_back_question_gen_chain | retriever,# The context is returned by the retriver after feeding to it the step-back question
        "question": RunnablePassthrough(),# This is the original user question
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

user_question = "Can you give me some tips for a trip to Brighton?"

answer = step_back_question_rag_chain.invoke(user_question)
print(answer)
#endregion main 3-Multiple query generation with MultiQueryRetriever

#region main 4-Hypotetical DocumentEmbeddings (HyDE)

# As discussed, embedding hypothetical questions can enhance RAG retrieval by 
# indexing document chunks with additional embeddings that represent questions answerable by 
# the content in each chunk. This approach makes embeddings of these hypothetical questions more 
# semantically similar to the user’s question than embeddings of the raw chunk text alone.

# A similar effect can be achieved with Hypothetical Document Embeddings (HyDE), a technique 
# that keeps the original chunk embeddings unchanged while generating hypothetical documents based on 
# the user’s question. The HyDE technique is shown in figure 9.5. In this approach, the LLM generates 
# hypothetical documents that would answer the user’s question. Rather than querying the document 
# store with the user’s original question, these generated documents are used. Because these hypothetical
#  documents are semantically closer to the document chunk text, they improve the relevance of retrieved 
#  content.

### Step 1: Setting up the chain to generate the hypotetical document associated to the user question
hyde_prompt_template = """
Write one sentence that could answer the provided question. 
Do not add anything else.
Question: {question}
Sentence:
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()

### Step 2: Testing the hyde generation chain
user_question = "What are the best beaches in Cornwall?"
hypotetical_document = hyde_chain.invoke(user_question)
hypotetical_document

### Step 3: Incorporating hyde chain into the RAG chain
retriever = uk_granular_chormaDb_collection.as_retriever()

rag_prompt_template = """
Given a question and some context, answer the question.
Only use the provided context to answer the question.
If you do not know the answer, just say I do not know. 

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template) 

hyde_rag_chain = (
    {
        "context": {"question": RunnablePassthrough()} 
           | hyde_chain | retriever,# The context is returned by the retriver after feeding to it the hypotetical document
        "question": RunnablePassthrough(),# This is the original user question
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

user_question = "What are the best beaches in Cornwall?"

answer = hyde_rag_chain.invoke(user_question)
print(answer)

user_question = "What are the best beaches in Cornwall?"

hypotetical_document = hyde_chain.invoke(user_question)
hypotetical_document

hyde_prompt_template = """
Write one sentence that could answer the provided question. 
Do not add anything else.
Question: {question}
Sentence:
"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()

#endregion main 4-Hypotetical DocumentEmbeddings (HyDE)