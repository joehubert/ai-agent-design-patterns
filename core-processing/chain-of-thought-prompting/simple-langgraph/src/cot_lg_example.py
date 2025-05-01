"""
Chain-of-Thought Pattern Implementation using LangGraph with Llama 3.2 via Ollama

This example demonstrates how to implement the Chain-of-Thought pattern using LangGraph
with Llama 3.2 served locally through Ollama, guiding the LLM to show its reasoning
process step-by-step for complex problems.
"""

import os
from typing import Dict, List, Annotated, TypedDict, Sequence, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END


# Define the state structure for our reasoning chain
class ChainOfThoughtState(TypedDict):
    problem: str
    reasoning_steps: List[str]
    intermediate_conclusions: List[str]
    final_answer: str


# Initialize the Llama 3.2 model via Ollama
def create_llm(temperature=0):
    return Ollama(
        model="llama3.2",  # Make sure you have this model pulled in Ollama
        temperature=temperature,
    )


# Define the system prompt that instructs the model to use chain-of-thought reasoning
SYSTEM_PROMPT = """You are an expert problem solver that carefully works through problems step-by-step.
When given a problem, follow these guidelines:

1. Analyze the problem carefully
2. Break it down into logical steps
3. Work through each step sequentially
4. Clearly state any intermediate conclusions
5. Draw a final conclusion based on your reasoning

Structure your response to clearly show each step of your reasoning process.
"""


# Step 1: Generate initial reasoning steps
def generate_reasoning_steps(state: ChainOfThoughtState) -> Dict:
    llm = create_llm()

    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    Problem: {state['problem']}
    
    Think step-by-step to solve this problem. First, identify the key components and initial reasoning steps needed to approach this problem.
    """

    response = llm.invoke(full_prompt)

    # Update the state with the reasoning steps
    return {"reasoning_steps": [response]}


# Step 2: Generate intermediate conclusions
def generate_intermediate_conclusions(state: ChainOfThoughtState) -> Dict:
    llm = create_llm()

    # Combine all the reasoning steps into context
    reasoning_context = "\n".join(state["reasoning_steps"])

    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    Problem: {state['problem']}
    
    Based on the following reasoning steps:
    
    {reasoning_context}
    
    Draw some intermediate conclusions that will help move toward a final answer.
    """

    response = llm.invoke(full_prompt)

    # Update the state with intermediate conclusions
    return {"intermediate_conclusions": [response]}


# Step 3: Generate final answer
def generate_final_answer(state: ChainOfThoughtState) -> Dict:
    llm = create_llm()

    # Combine all the context from previous steps
    reasoning_context = "\n".join(state["reasoning_steps"])
    conclusions_context = "\n".join(state["intermediate_conclusions"])

    full_prompt = f"""
    {SYSTEM_PROMPT}
    
    Problem: {state['problem']}
    
    Reasoning steps:
    {reasoning_context}
    
    Intermediate conclusions:
    {conclusions_context}
    
    Based on the above reasoning process, provide a final answer to the problem.
    """

    response = llm.invoke(full_prompt)

    # Update the state with the final answer
    return {"final_answer": response}


# Create the Chain-of-Thought graph
def create_cot_graph():
    # Initialize the graph with our state
    workflow = StateGraph(ChainOfThoughtState)

    # Add nodes for each step in the chain-of-thought process
    workflow.add_node("generate_reasoning", generate_reasoning_steps)
    workflow.add_node("generate_conclusions", generate_intermediate_conclusions)
    workflow.add_node("generate_answer", generate_final_answer)

    # Define the edges between nodes (the reasoning chain flow)
    workflow.add_edge("generate_reasoning", "generate_conclusions")
    workflow.add_edge("generate_conclusions", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Set the entry point
    workflow.set_entry_point("generate_reasoning")

    # Compile the graph
    return workflow.compile()


# Example usage function
def solve_problem_with_cot(problem: str):
    # Create the chain-of-thought graph
    cot_graph = create_cot_graph()

    # Initialize the state with the problem
    initial_state = {
        "problem": problem,
        "reasoning_steps": [],
        "intermediate_conclusions": [],
        "final_answer": "",
    }

    # Execute the graph
    result = cot_graph.invoke(initial_state)

    return result


graph = create_cot_graph()

# Example usage
if __name__ == "__main__":
    # Example math problem
    problem = "Sarah has twice as many marbles as John. John has 5 more marbles than Lisa. If Lisa has 15 marbles, how many marbles do they have in total?"

    # Run the chain-of-thought reasoning
    result = solve_problem_with_cot(problem)

    # Print the full reasoning process
    print("\n=== PROBLEM ===")
    print(result["problem"])

    print("\n=== REASONING STEPS ===")
    for i, step in enumerate(result["reasoning_steps"]):
        print(f"Step {i+1}:")
        print(step)
        print("-" * 50)

    print("\n=== INTERMEDIATE CONCLUSIONS ===")
    for i, conclusion in enumerate(result["intermediate_conclusions"]):
        print(f"Conclusion {i+1}:")
        print(conclusion)
        print("-" * 50)

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
