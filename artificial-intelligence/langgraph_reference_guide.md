# LangGraph & LangChain Quick Reference Guide

## Core Packages Overview

| Package | Purpose | When to Use |
|---------|---------|-------------|
| **langgraph** | Build multi-step AI workflows with state management | When you need agents to make decisions, loop through steps, or maintain context across multiple operations |
| **langchain-openai** | Connect to OpenAI-compatible APIs (including Gemini) | When you need to call LLMs for reasoning, text generation, or analysis |
| **langchain-groq** | Connect to Groq's fast inference API | When you need extremely fast LLM responses or want a reliable fallback |
| **langchain-core** | Core abstractions used by all LangChain packages | Automatically installed with other packages - provides base classes |

---

## LangGraph Components

### 1. State Management

| Component | What It Does | Example Use |
|-----------|--------------|-------------|
| `TypedDict` | Defines the structure of data flowing through your workflow | Specifying what data each step receives and can modify |
| `StateGraph(StateType)` | Creates a workflow graph that maintains state between steps | Building an agent that remembers previous steps' outputs |
| State object | Dictionary passed between nodes, gets updated at each step | Storing weather data, analysis results, recommendations |

**Key Concept:** State is like a shared notebook that each agent step can read from and write to.

---

### 2. Graph Building

| Function/Method | What It Does | Code Example |
|-----------------|--------------|--------------|
| `workflow.add_node(name, function)` | Adds a step to your workflow | `workflow.add_node("analyze", analyzer_node)` |
| `workflow.set_entry_point(name)` | Defines where the workflow starts | `workflow.set_entry_point("fetch")` |
| `workflow.add_edge(from, to)` | Creates a direct connection between steps | `workflow.add_edge("analyze", "recommend")` |
| `workflow.add_conditional_edges(from, condition_func, mapping)` | Routes to different steps based on logic | If data fetch fails, skip to end; otherwise continue |
| `workflow.compile()` | Turns your graph definition into an executable agent | `agent = workflow.compile()` |
| `END` | Special marker indicating workflow completion | `workflow.add_edge("recommend", END)` |

**Key Concept:** Think of it like building a flowchart where each box is a function, and you define the arrows between them.

---

### 3. Node Functions (The Actual Work)

| Pattern | Structure | Purpose |
|---------|-----------|---------|
| Node signature | `def node_name(state: StateType) -> StateType:` | Every node takes current state, returns updated state |
| State update | `return {**state, 'new_key': new_value}` | Preserve existing state, add/update specific fields |
| Pure Python logic | Any Python code inside nodes | Do calculations, call APIs, process data without LLM |
| LLM calls | `llm.invoke([HumanMessage(content=prompt)])` | Ask LLM to reason about or generate text |

**Key Concept:** Nodes are just Python functions that receive state, do work, and return updated state.

---

## LangChain Components

### 4. LLM Integration

| Class | What It Does | Configuration Options |
|-------|--------------|----------------------|
| `ChatOpenAI` | Connects to OpenAI-compatible APIs | `model`: which LLM<br>`temperature`: creativity (0-1)<br>`api_key`: authentication<br>`base_url`: API endpoint |
| `ChatGroq` | Connects to Groq's API | `model`: which Groq model<br>`temperature`: creativity<br>`api_key`: Groq key |
| `.invoke(messages)` | Sends messages to LLM and gets response | Takes list of message objects, returns response |

**Key Concept:** These are wrappers that make it easy to talk to different LLM providers with the same code structure.

---

### 5. Message Types

| Type | Purpose | Example |
|------|---------|---------|
| `HumanMessage(content="...")` | Represents user/system input to LLM | `HumanMessage(content="Analyze this data: ...")` |
| `SystemMessage(content="...")` | Sets LLM behavior/role | `SystemMessage(content="You are a data analyst")` |
| `response.content` | Extracts text from LLM response | `analysis = response.content` |

**Key Concept:** Messages are structured ways to communicate with LLMs - like formatting an email vs plain text.

---

## Workflow Execution Flow

| Step | What Happens | Code |
|------|--------------|------|
| 1. Define state structure | Create TypedDict with all fields your workflow needs | `class WeatherState(TypedDict): ...` |
| 2. Create graph | Initialize StateGraph with your state type | `workflow = StateGraph(WeatherState)` |
| 3. Add nodes | Register all functions that will process state | `workflow.add_node("fetch", fetch_function)` |
| 4. Connect nodes | Define how data flows between steps | `workflow.add_edge("fetch", "analyze")` |
| 5. Compile | Turn definition into executable | `agent = workflow.compile()` |
| 6. Run | Execute with initial state | `result = agent.invoke(initial_state)` |

---

## Key Design Patterns

### Pattern 1: Tool-Then-LLM
```python
def analyzer_node(state):
    # 1. Use Python to crunch numbers
    stats = calculate_statistics(state['data'])
    
    # 2. Use LLM to interpret results
    llm = get_llm()
    analysis = llm.invoke([HumanMessage(f"Interpret: {stats}")])
    
    return {**state, 'analysis': analysis.content}
```
**Why:** LLMs are expensive/slow - use Python for computation, LLMs for reasoning.

---

### Pattern 2: Conditional Routing
```python
def should_continue(state) -> Literal["next", "end"]:
    if state['has_error']:
        return "end"
    return "next"

workflow.add_conditional_edges("check", should_continue, {
    "next": "process",
    "end": END
})
```
**Why:** Agents need to make decisions based on data, not just follow linear paths.

---

### Pattern 3: Multi-Step Enrichment
```python
# Each node adds to the state
def step1(state):
    return {**state, 'data': fetch_data()}

def step2(state):
    return {**state, 'processed': process(state['data'])}

def step3(state):
    return {**state, 'result': analyze(state['processed'])}
```
**Why:** Complex workflows are easier to understand and debug when broken into focused steps.

---

## Common "Gotchas" & Tips

| Issue | Why It Happens | Solution |
|-------|----------------|----------|
| API key not found | Environment variables aren't set in current shell | Set in same terminal you run script, or hardcode in Python temporarily |
| State not updating | Forgot to return updated state from node | Always return `{**state, 'new_field': value}` |
| Wrong LLM response format | Prompt wasn't clear enough | Be explicit: "Respond with JSON" or "Answer in 2 sentences" |
| Workflow hangs | Conditional edge has no matching case | Ensure all possible return values map to nodes |
| Import errors | Wrong package installed | Use `langchain-openai` not `openai`, `langchain-groq` not `groq` |

---

## Extending to Data Analysis

To adapt this for CSV analysis, replace:

| Weather Example | Data Analysis Equivalent |
|-----------------|-------------------------|
| `fetch_weather_data()` | `pd.read_csv(file)` |
| `calculate_comfort_score()` | `df.describe()`, `df.corr()`, custom metrics |
| Weather analysis prompt | "Analyze these statistics and identify notable patterns" |
| Recommendations | "Suggest data quality improvements" or "Recommend visualizations" |

**The structure stays the same** - just swap out the domain-specific logic!

---

## Quick Troubleshooting Commands

```bash
# Check installed packages
pip list | grep langchain
pip list | grep langgraph

# Verify API key is set (PowerShell)
echo $env:GOOGLE_API_KEY

# Test LLM connection directly
python -c "from langchain_openai import ChatOpenAI; import os; llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url='https://generativelanguage.googleapis.com/v1beta/openai/', model='gemini-2.0-flash-exp'); print(llm.invoke('Hi').content)"
```

---

## Resources for Deep Dives

| Topic | Where to Learn More |
|-------|-------------------|
| LangGraph concepts | https://langchain-ai.github.io/langgraph/ |
| LangChain integrations | https://python.langchain.com/docs/integrations/platforms/ |
| State management patterns | Search: "LangGraph state management" |
| Gemini API docs | https://ai.google.dev/gemini-api/docs |
| Groq documentation | https://console.groq.com/docs |