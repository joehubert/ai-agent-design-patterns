from langgraph.graph import StateGraph
from typing import Dict, TypedDict, List
import operator

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import re
import time


# Define our state
class State(TypedDict):
    query: str
    requires_rag: bool
    context: List[str]
    reformulated_query: str
    response: str
    debug_info: Dict


# Create the LLM instance using Ollama with Llama 3.2
llm = Ollama(
    model="llama3.2",
    temperature=0.1,  # Slightly higher temperature for more creative outputs
    base_url="http://localhost:11434",  # Default Ollama URL
    num_ctx=4096,  # Increase context window
)

# Create the graph
graph = StateGraph(State)


def initialize_vector_store():
    # Load the document
    loader = TextLoader("../docs/school_events.md")
    documents = loader.load()

    # Split by headers first, then content with appropriate chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks to capture more context
        chunk_overlap=200,
        separators=["## ", "### ", "\n\n", "\n", " ", ""],  # Respect markdown structure
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store with Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store, chunks


# Initialize the vector store
vector_store, all_chunks = initialize_vector_store()


# Node 1: Determine if RAG is needed
def determine_if_rag_needed(state: State) -> State:
    query = state["query"]

    # Enhanced system prompt with better classification criteria
    llm_response = llm.invoke(
        f"""<instruction>
You are an expert assistant that determines if queries need specific information about Oakridge High School events.

Given this query: "{query}"

Your task is to determine if this query is asking for SPECIFIC information about an Oakridge High School event. 

IMPORTANT CLASSIFICATION RULES:
1. ANY query asking about WHEN, WHERE, WHO, or HOW MUCH regarding a specific named event MUST be classified as TRUE
2. Questions about times, dates, locations, costs, people, or other specific details of events are TRUE
3. When uncertain, classify as TRUE rather than FALSE

Examples that NEED specific information (answer TRUE):
- When is the Battle of the Bands event?
- What time does the Battle of the Bands start?
- What bands are performing at the Battle of the Bands?
- Who should I contact about the Spring Talent Show?
- How much do tickets cost for the graduation ceremony?
- What food will be available at Field Day?

Examples that do NOT need specific information (answer FALSE):
- What is a talent show in general?
- How do battle of the bands competitions usually work in high schools?
- What are good talents to showcase at a high school event?
- What types of food are normally served at school events?

Importantly, anything asking WHEN or WHAT TIME regarding a specific event MUST be TRUE.

Return ONLY the word "TRUE" or "FALSE" with no explanation.
</instruction>"""
    )

    # Parse the response more carefully
    response_text = llm_response.strip().upper()
    # Check for TRUE in any form (TRUE, True, true)
    requires_rag = "TRUE" in response_text

    # Force RAG for time-related queries about specific events (as a backup)
    lower_query = query.lower()
    if (
        "time" in lower_query or "when" in lower_query or "start" in lower_query
    ) and "battle of the bands" in lower_query:
        requires_rag = True

    return {
        "query": query,
        "requires_rag": requires_rag,
        "context": [],
        "reformulated_query": "",
        "response": "",
        "debug_info": {
            "rag_decision_raw": llm_response,
            "final_decision": requires_rag,
        },
    }


# Node 2: Reformulate the query for better retrieval
def reformulate_query(state: State) -> State:
    if not state["requires_rag"]:
        return state

    query = state["query"]

    # Use LLM to expand the query for better retrieval
    expanded_query = llm.invoke(
        f"""<instruction>
You are an expert at information retrieval for school events.

Original query: "{query}"

Rewrite this query to include potential keywords related to Oakridge High School events that would help in finding relevant information. Consider including terms like: date, time, location, rules, cost, contact, participants, requirements, food, etc. that might be in the document.

The goal is to create an expanded search query that will help find the most relevant information in a document about school events.

Return ONLY the expanded query with no additional explanation.
</instruction>"""
    )

    # Clean the expanded query
    expanded_query = expanded_query.strip()

    return {**state, "reformulated_query": expanded_query}


# Node 3: Retrieve relevant documents if needed
def retrieve_documents(state: State) -> State:
    if not state["requires_rag"]:
        return state

    # Use the reformulated query if available, otherwise use original
    query = state["reformulated_query"] or state["query"]

    # Perform vector search with more results
    documents = vector_store.similarity_search(query, k=5)
    context = [doc.page_content for doc in documents]

    # Format the context for better readability by the model
    formatted_context = "\n\n---\n\n".join(context)

    debug_info = state.get("debug_info", {})
    debug_info["retrieved_docs_count"] = len(documents)
    debug_info["reformulated_query"] = query

    return {**state, "context": context, "debug_info": debug_info}


# Node 4: Generate response
def generate_response(state: State) -> State:
    query = state["query"]

    if state["requires_rag"]:
        context = state["context"]

        # More explicit system prompt for RAG response
        prompt = f"""<instruction>
You are an expert assistant for Oakridge High School events. You have access to information about various school events and activities.

USER QUERY: {query}

RELEVANT INFORMATION FROM SCHOOL DOCUMENTS:

{context}

Answer the user's query based ONLY on the information provided above. Be specific and provide details directly from the documents when possible. Include dates, times, locations, contacts, or other specific details when they're available in the given information.

If the specific information needed is not in the provided documents, state clearly that you don't have that specific information, but you can share what you do know from the documents.

Respond in a helpful, conversational tone suitable for a school community member.
</instruction>"""
    else:
        # More explicit system prompt for general knowledge response
        prompt = f"""<instruction>
You are an expert assistant for general questions about school events. 

USER QUERY: {query}

This query doesn't require specific information about Oakridge High School events, so answer based on your general knowledge about school events, activities, and best practices.

Provide a helpful, informative answer without making up specific details about Oakridge High School events (like dates, contacts, prices, etc.) that would only be found in official school documents.

Respond in a helpful, conversational tone suitable for a school community member.
</instruction>"""

    # Add a retry mechanism for LLM calls
    max_attempts = 3
    response = None

    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                response = f"Error generating response: {str(e)}"
            else:
                time.sleep(2)  # Wait before retrying

    debug_info = state.get("debug_info", {})
    debug_info["response_type"] = (
        "RAG" if state["requires_rag"] else "General Knowledge"
    )

    return {**state, "response": response, "debug_info": debug_info}


# Add nodes to graph
graph.add_node("determine_if_rag_needed", determine_if_rag_needed)
graph.add_node("reformulate_query", reformulate_query)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("generate_response", generate_response)


# Define the conditional edge
def route_based_on_rag_need(state):
    if state["requires_rag"]:
        return "reformulate_query"
    else:
        return "generate_response"


# Add edges
graph.add_conditional_edges(
    "determine_if_rag_needed",
    route_based_on_rag_need,
    {
        "reformulate_query": "reformulate_query",
        "generate_response": "generate_response",
    },
)
graph.add_edge("reformulate_query", "retrieve_documents")
graph.add_edge("retrieve_documents", "generate_response")

# Set the entry point
graph.set_entry_point("determine_if_rag_needed")

# Compile the graph
app = graph.compile()


# Example usage with debug output
def process_query(query: str, debug=False):
    result = app.invoke(
        {
            "query": query,
            "requires_rag": False,
            "context": [],
            "reformulated_query": "",
            "response": "",
            "debug_info": {},
        }
    )

    if debug:
        print("\n=== DEBUG INFO ===")
        for key, value in result["debug_info"].items():
            print(f"{key}: {value}")
        print("=================\n")

    return result["response"]


# Example code to run this directly
if __name__ == "__main__":
    # Test queries
    queries = [
        "When is the Battle of the Bands event?",
        "What bands are playing at the Battle of the Bands?",
        "Who is in the band Cosmic Decay?",
        "What food will be available at the graduation ceremony?",
        "What is a talent show?",
        "How do battle of the bands competitions usually work?",
    ]

    for query in queries:
        print(f"\n==== Query: {query} ====")
        response = process_query(query, debug=True)
        print(f"Response: {response}")
        print("=" * 50)
