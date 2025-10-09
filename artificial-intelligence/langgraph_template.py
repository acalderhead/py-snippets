#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────────────────────────
# Module Documentation
# ─────────────────────────────────────────────────────────────────────────────
"""
LangGraph Agent Workflow Template
─────────────────────────────────
Purpose
    A generalizable template for building agentic workflows using LangGraph.
    Demonstrates common patterns: state management, conditional routing, 
    LLM integration, and tool usage.

Context
    LangGraph is a library for building stateful, multi-actor applications
    with LLMs. It extends LangChain with graph-based workflow orchestration,
    enabling complex agent behaviors with cycles, conditional edges, and
    persistent state.

Key Concepts
    - **State**: TypedDict that flows through the graph
    - **Nodes**: Functions that process state and return updates
    - **Edges**: Connections between nodes (linear or conditional)
    - **Graph**: Compiled workflow that orchestrates execution

Inputs
    Initial state dictionary matching your StateGraph schema.

Outputs
    Final state after all nodes execute, containing accumulated results.

Usage
─────
    python langgraph_template.py
    
    # Or import and use programmatically:
    from langgraph_template import create_agent, run_agent
    agent = create_agent()
    result = agent.invoke(initial_state)

Dependencies
────────────────────────
- langgraph >= 0.2.0
- langchain-core >= 0.3.0
- langchain-openai or langchain-groq (for LLM providers)

Documentation
─────────────
Official LangGraph Docs: https://langchain-ai.github.io/langgraph/
LangChain Docs: https://python.langchain.com/docs/
Tutorials: https://langchain-ai.github.io/langgraph/tutorials/

Limitations
───────────
- Requires API keys for LLM providers (set as environment variables)
- State must be serializable (JSON-compatible types)
"""

__author__  = "Aidan Calderhead"
__created__ = "2025-10-09"
__version__ = "1.0.0"
__license__ = "MIT"

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import logging
import sys
from typing   import TypedDict, Literal, Annotated, Any
from operator import add

# LangGraph imports for graph construction
from langgraph.graph import StateGraph, END, START

# LangChain imports for LLM integration
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai        import ChatOpenAI
from langchain_groq          import ChatGroq

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_TEMPERATURE:  float = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The state schema that flows through the graph.
    
    State Design Patterns:
    ──────────────────────
    1. **Replacement** (default): Each node returns full updates that replace
       existing values.
       
    2. **Accumulation**: Use Annotated[list, add] to append to lists rather
       than replace them. Useful for message histories or step tracking.
       
    3. **Custom Reducers**: Define custom functions to merge state updates
       (e.g., merging dicts, computing aggregates).
    
    Example with accumulation:
        messages: Annotated[list, add]  # Appends instead of replacing
    """
    # Input parameters
    input_data:   str
    llm_provider: str
    
    # Processing results (replaced on each update)
    processed_data:  dict
    analysis_result: str
    final_output:    str
    
    # Accumulated data (appends on each update)
    execution_log: Annotated[list[str], add]
    
    # Control flow
    should_retry:  bool
    error_message: str

# ─────────────────────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(
    provider:    str   = DEFAULT_LLM_PROVIDER, 
    temperature: float = DEFAULT_TEMPERATURE
):
    """
    Initialize LLM provider.
    
    Supported Providers:
    ────────────────────
    - gemini: Google's Gemini models via OpenAI-compatible API
    - groq: Groq's fast inference (Llama models)
    - openai: OpenAI's GPT models
    
    Environment Variables Required:
    ───────────────────────────────
    - GOOGLE_API_KEY (for Gemini)
    - GROQ_API_KEY (for Groq)
    - OPENAI_API_KEY (for OpenAI)
    
    Returns:
    ────────
    langchain_core.language_models.chat_models.BaseChatModel
        Initialized LLM instance
    """
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        return ChatOpenAI(
            model       = "gemini-2.0-flash-exp",
            temperature = temperature,
            api_key=api_key,
            base_url    = (
              "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
        )
    
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        return ChatGroq(
            model       = "llama-3.3-70b-versatile",
            temperature = temperature,
            api_key     = api_key
        )
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        return ChatOpenAI(
            model       = "gpt-4o-mini",
            temperature = temperature,
            api_key     = api_key
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# TOOL/UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def example_data_processor(data: str) -> dict:
    """
    Example Python tool function.
    
    Tools are regular Python functions that perform specific tasks:
    - API calls
    - Data processing
    - File I/O
    - Calculations
    
    They should be pure functions when possible (no side effects).
    """
    return {
        "processed":  data.upper(),
        "length":     len(data),
        "word_count": len(data.split())
    }


def example_api_call(query: str) -> dict:
    """
    Example external API integration.
    
    Pattern for API calls:
    ──────────────────────
    1. Add error handling (try/except)
    2. Set timeouts
    3. Return consistent structure (success/error dict)
    4. Log important events
    """
    try:
        # Simulate API call
        logger.info(f"Making API call with query: {query}")
        return {"status": "success", "data": f"Result for {query}"}
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {"status": "error", "message": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────────────────────────────────────

def input_processor_node(state: AgentState) -> dict:
    """
    Node Pattern 1: Basic Processing Node
    
    Purpose:
        Process input data using Python tools (no LLM).
    
    Node Design:
    ────────────
    - Takes state as input
    - Performs computation/processing
    - Returns dict with state updates
    - Logs progress for debugging
    
    Return Value:
    ─────────────
    Dict containing only the fields you want to update. LangGraph
    automatically merges this with existing state.
    """
    logger.info(f"Processing input: {state['input_data']}")
    
    processed = example_data_processor(state['input_data'])
    
    return {
        "processed_data": processed,
        "execution_log":  ["Input processed successfully"]
    }


def llm_analyzer_node(state: AgentState) -> dict:
    """
    Node Pattern 2: LLM Integration Node
    
    Purpose:
        Use LLM to analyze, summarize, or generate content.
    
    LLM Invocation Patterns:
    ────────────────────────
    1. Single message: llm.invoke([HumanMessage(content="...")])
    2. System + User: llm.invoke([SystemMessage(...), HumanMessage(...)])
    3. Conversation: llm.invoke([msg1, msg2, msg3, ...])
    
    Prompt Engineering Tips:
    ────────────────────────
    - Be specific about format and constraints
    - Provide examples for complex tasks
    - Use system messages for role/behavior instructions
    """
    logger.info("Analyzing with LLM...")
    
    llm = get_llm(state['llm_provider'])
    
    # Construct prompt with context
    prompt = f"""Analyze the following processed data:

Data: {state['processed_data']}

Provide a brief analysis (2-3 sentences) highlighting key insights."""

    # Invoke LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    
    logger.info("Analysis complete")
    
    return {
        "analysis_result": analysis,
        "execution_log":   ["LLM analysis completed"]
    }


def decision_maker_node(state: AgentState) -> dict:
    """
    Node Pattern 3: Decision/Routing Node
    
    Purpose:
        Make decisions that affect graph routing (used with conditional edges).
    
    Pattern:
    ────────
    Set state fields that conditional edge functions will check.
    Don't return routing decisions directly - that's done in the
    conditional edge function.
    """
    logger.info("Making routing decision...")
    
    # Example: Decide if we need to retry based on data quality
    data_length  = state['processed_data'].get('length', 0)
    should_retry = data_length < 5
    
    if should_retry:
        error_msg = "Input too short, retry needed"
    else:
        error_msg = ""
    
    return {
        "should_retry":  should_retry,
        "error_message": error_msg,
        "execution_log": [f"Decision: retry={should_retry}"]
    }


def output_generator_node(state: AgentState) -> dict:
    """
    Node Pattern 4: Final Output Node
    
    Purpose:
        Generate final results, often combining multiple state elements.
    
    Common Use Cases:
    ─────────────────
    - Format final response
    - Generate reports
    - Prepare data for external systems
    """
    logger.info("Generating final output...")
    
    llm = get_llm(state['llm_provider'])
    
    prompt = f"""Based on this analysis:

{state['analysis_result']}

Generate 3 actionable recommendations. Be specific and practical."""

    response = llm.invoke([HumanMessage(content = prompt)])
    
    return {
        "final_output":  response.content,
        "execution_log": ["Final output generated"]
    }


def error_handler_node(state: AgentState) -> dict:
    """
    Node Pattern 5: Error Handling Node
    
    Purpose:
        Handle errors gracefully and provide fallback behavior.
    
    Pattern:
    ────────
    Reached via conditional routing when errors occur.
    """
    logger.warning(f"Error handling triggered: {state['error_message']}")
    
    return {
        "final_output":  (
            f"Could not complete processing: {state['error_message']}"
        ),
        "execution_log": ["Error handled"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def route_after_decision(
    state: AgentState
) -> Literal["generate_output", "handle_error"]:
    """
    Conditional Edge Pattern: Route based on state
    
    Purpose:
        Determine next node based on state values.
    
    Returns:
    ────────
    String matching one of the node names in add_conditional_edges mapping.
    Can also return END to terminate.
    
    Usage in Graph:
    ───────────────
    workflow.add_conditional_edges(
        "decision_node",
        route_after_decision,
        {
            "generate_output": "output_node",
            "handle_error": "error_node"
        }
    )
    """
    if state['should_retry']:
        logger.info("Routing to error handler")
        return "handle_error"
    else:
        logger.info("Routing to output generator")
        return "generate_output"


def should_continue_processing(
    state: AgentState
) -> Literal["analyze", "end"]:
    """
    Conditional Edge Pattern: Early termination
    
    Purpose:
        Skip processing if prerequisites aren't met.
    
    Common Use Cases:
    ─────────────────
    - Data validation failed
    - API call returned error
    - User-specified conditions not met
    """
    if not state.get('processed_data'):
        logger.warning("No processed data, ending early")
        return "end"
    return "analyze"

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_agent() -> Any:
    """
    Build the state graph.
    
    Graph Construction Steps:
    ─────────────────────────
    1. Initialize StateGraph with state schema
    2. Add nodes (functions that process state)
    3. Set entry point
    4. Add edges (linear or conditional)
    5. Compile the graph
    
    Edge Types:
    ───────────
    - add_edge(from, to): Always goes from -> to
    - add_conditional_edges(from, condition_func, mapping): 
          Routes based on condition
    - set_entry_point(node): Where execution starts
    - END: Special terminal node
    
    Common Patterns:
    ────────────────
    Linear: A -> B -> C -> END
    Branching: A -> (B or C) -> D -> END
    Cycles: A -> B -> C -> B (loop back)
    """
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("process_input",   input_processor_node)
    workflow.add_node("make_decision",   decision_maker_node)
    workflow.add_node("analyze",         llm_analyzer_node)
    workflow.add_node("generate_output", output_generator_node)
    workflow.add_node("handle_error",    error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    # Add edges
    # Linear edge: always goes from process_input to make_decision
    workflow.add_edge("process_input", "make_decision")
    
    # Conditional edge: route based on decision
    workflow.add_conditional_edges(
        "make_decision",
        route_after_decision,
        {
            "generate_output": "analyze",      # If no retry, go to analyze
            "handle_error":    "handle_error"  # If retry, go to error handler
        }
    )
    
    # Continue linear flow
    workflow.add_edge("analyze", "generate_output")
    
    # Both paths end here
    workflow.add_edge("generate_output", END)
    workflow.add_edge("handle_error",    END)
    
    # Compile the graph
    return workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    input_data:   str,
    llm_provider: str = DEFAULT_LLM_PROVIDER
) -> dict:
    """
    Execute the agent workflow.
    
    Execution Patterns:
    ───────────────────
    1. invoke(): Run once with initial state, return final state
    2. stream(): Yield state after each node execution
    3. astream(): Async version of stream()
    
    Args:
        input_data: The data to process
        llm_provider: Which LLM to use (gemini/groq/openai)
    
    Returns:
        Final state dictionary after all nodes execute
    """
    logger.info(F"STARTING {llm_provider} AGENT EXECUTION ON {input_data}")
    
    agent = create_agent()
    
    # Initialize state
    initial_state = {
        "input_data":      input_data,
        "llm_provider":    llm_provider,
        "processed_data":  {},
        "analysis_result": "",
        "final_output":    "",
        "execution_log":   [],
        "should_retry":    False,
        "error_message":   ""
    }
    
    # Execute graph
    final_state = agent.invoke(initial_state)
    
    # Display results
    logger.info(f"EXECUTION COMPLETE: {final_state['final_output']}")
    logger.info(f"\nExecution Log:\n" + "\n".join(final_state['execution_log']))
    
    return final_state


def stream_agent(
    input_data:   str,
    llm_provider: str = DEFAULT_LLM_PROVIDER
) -> None:
    """
    Stream agent execution (see state after each node).
    
    Streaming Pattern:
    ──────────────────
    Use when you want to show progress or handle partial results.
    Each iteration yields (node_name, state_update).
    """
    agent = create_agent()
    
    initial_state = {
        "input_data":      input_data,
        "llm_provider":    llm_provider,
        "processed_data":  {},
        "analysis_result": "",
        "final_output":    "",
        "execution_log":   [],
        "should_retry":    False,
        "error_message":   ""
    }
    
    logger.info("Starting streaming execution...\n")
    
    for step in agent.stream(initial_state):
        node_name    = list(step.keys())[0]
        state_update = step[node_name]
        logger.info(f"Completed node: {node_name}")
        logger.info(f"State update: {list(state_update.keys())}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run example agent execution."""
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning(
          "GOOGLE_API_KEY not set! Set with: export GOOGLE_API_KEY=your-key"
        )
    
    # Example 1: Standard execution
    logger.info("\n### EXAMPLE 1: Standard Execution ###\n")
    result = run_agent(
        input_data="Climate change impact on agriculture",
        llm_provider="gemini"
    )
    
    # Example 2: Streaming execution
    logger.info("\n### EXAMPLE 2: Streaming Execution ###\n")
    stream_agent(
        input_data="Renewable energy adoption trends",
        llm_provider="gemini"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
