# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aitor is a Python framework for building intelligent, memory-enabled agents that can execute complex workflows. The framework provides:

- **Memory-Enabled Agents**: Stateful agents (`Aitor`) with typed memory management
- **DAG Workflows**: Task dependency management through `Aitorflow` class
- **Async Processing**: Thread-safe execution with both blocking (`ask`) and non-blocking (`tell`) APIs
- **Task Chaining**: Intuitive `>>` operator for defining task dependencies

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Running Examples
```bash
# Basic workflow example
python example.py

# Aitor agent example
python aitor_example.py
```

### Testing
Currently no test framework is configured. Tests directory exists but is empty.

### Building and Distribution
```bash
# Build package
python -m build

# Install locally
pip install -e .
```

## Architecture

### Core Components

1. **`Aitor` class** (`src/aitor/aitor.py`):
   - Generic class typed by memory structure: `Aitor[T]`
   - Thread-safe memory access with `threading.Lock`
   - Class-level `ThreadPoolExecutor` for concurrent execution
   - Per-thread event loops for async operations

2. **`Task` class** (`src/aitor/task.py`):
   - Wraps Python functions as workflow tasks
   - Supports `>>` operator for dependency chaining
   - Maintains upstream/downstream task relationships
   - Can be created via `@task` decorator

3. **`Aitorflow` class** (`src/aitor/aitorflows.py`):
   - DAG-based workflow execution engine
   - Validates workflow structure and prevents cycles
   - Executes tasks in parallel using `ThreadPoolExecutor`
   - Provides Mermaid diagram visualization

### Key Design Patterns

#### Memory Management
- All memory access is thread-safe using locks
- Memory is generic typed: `Aitor[List[str]]`, `Aitor[Dict[str, Any]]`
- Access via `get_memory()` and `set_memory()` methods

#### Task Execution Flow
```python
# Define tasks
@task
def process_text(text: str) -> str:
    return text.upper()

@task  
def count_words(text: str) -> int:
    return len(text.split())

# Chain tasks
process_text >> count_words

# Create workflow
workflow = Aitorflow()
workflow.add_task(process_text)  # Auto-discovers connected tasks
```

#### Aitor Usage Patterns
```python
# Create aitor with typed memory
aitor = Aitor[List[str]](initial_memory=[])

# Attach workflow
aitor.attach_workflow(workflow)

# Blocking execution
result = await aitor.ask("input message")

# Non-blocking execution  
aitor.tell("input message")

# Always shutdown
Aitor.shutdown()
```

### Threading Model

- **Class-level thread pool**: Shared `ThreadPoolExecutor` across all Aitor instances
- **Per-thread event loops**: Each thread gets its own event loop via `threading.local()`
- **Thread-safe memory**: All memory operations protected by locks
- **Async execution**: Tasks run in thread pool with async/await support

### Workflow Execution

1. **Validation**: Checks for cycles and parameter compatibility
2. **Topological execution**: Entry tasks → parallel execution → exit tasks
3. **Result aggregation**: Returns `Dict[str, Any]` mapping task names to results
4. **Error handling**: Fails fast on any task error

## Important Implementation Details

### Task Parameter Handling
- Single upstream task: `downstream.execute(result)`
- Multiple upstream tasks: `downstream.execute(*results)` (ordered by `upstream_order`)
- Tasks should handle variable arguments if receiving multiple inputs

### Event Loop Management
- Each thread gets its own event loop stored in `threading.local()`
- `get_loop()` creates new loops as needed
- Proper cleanup in `shutdown()` method

### Memory Persistence
- `persist_memory()` and `_load_memory()` methods are stubs for extension
- Current implementation uses in-memory storage only

## Development Guidelines

### When Adding New Features
- Maintain thread safety for all shared resources
- Use type hints consistently (project uses Generic[T] pattern)
- Add proper logging statements for debugging
- Consider both sync and async execution contexts

### Error Handling
- Workflow validation happens before execution
- Task failures propagate up and stop workflow execution
- Use descriptive error messages with task names

### Performance Considerations
- Thread pool size is auto-configured based on CPU count
- Tasks execute in parallel where dependencies allow
- Memory operations are locked but should be brief

## Package Structure
```
src/aitor/
├── __init__.py          # Package exports
├── aitor.py            # Main Aitor agent class
├── aitorflows.py       # DAG workflow management
└── task.py             # Task wrapper and decorator
```

## ReAct Agent Framework

### Overview
The framework now includes a comprehensive ReAct (Reasoning and Acting) agent system that extends the base Aitor architecture. ReAct agents combine memory-enabled processing with intelligent reasoning and tool execution capabilities.

### Core ReAct Components

#### 1. ReactAgent (`src/aitor/react_agent.py`)
- **Purpose**: Main agent class that combines Aitor's memory system with ReAct reasoning
- **Key Features**:
  - Extends `Aitor[ReactMemory]` for typed memory management
  - Integrates reasoning engine for Think -> Act -> Observe loops
  - Tool registry for dynamic tool management
  - LLM integration for intelligent responses
  - Session management and memory persistence

#### 2. ReasoningEngine (`src/aitor/reasoning.py`)
- **Purpose**: Core ReAct reasoning loop implementation
- **Key Features**:
  - Orchestrates Think -> Act -> Observe cycles
  - Parses LLM responses into reasoning steps
  - Executes tool actions with error handling
  - Manages reasoning step limits and error recovery

#### 3. ToolRegistry (`src/aitor/tools.py`)
- **Purpose**: Centralized tool management system
- **Key Features**:
  - Dynamic tool registration and execution
  - Async tool execution with timeout handling
  - Parameter validation and schema generation
  - Thread-safe tool management

#### 4. ReactMemory (`src/aitor/memory.py`)
- **Purpose**: Structured memory for ReAct agents
- **Components**:
  - Conversation history with role-based messages
  - Tool execution history with results
  - Reasoning trace with step-by-step analysis
  - Context management with automatic pruning

#### 5. LLM Interface (`src/aitor/llm.py`)
- **Purpose**: Unified interface for multiple LLM providers
- **Supported Providers**:
  - OpenAI (via `openai` library)
  - Anthropic (via `anthropic` library)
  - Mock provider for testing
- **Features**: Connection pooling, retry logic, JSON response parsing

### ReAct Agent Usage Patterns

#### Basic Agent Creation
```python
from aitor import ReactAgent, create_react_agent

# Using convenience function
agent = await create_react_agent(
    name="MyAgent",
    llm_provider="openai",
    llm_config={"api_key": "...", "model": "gpt-4"},
    tools=[calculator_tool, search_tool]
)

# Using builder pattern
agent = await (ReactAgentBuilder()
               .name("MyAgent")
               .llm_manager(llm_manager)
               .max_reasoning_steps(20)
               .build())
```

#### Tool Registration
```python
# Register function as tool
@tool(name="calculator", description="Perform calculations")
def calculate(expression: str) -> float:
    return eval(expression)

await agent.register_tool(calculate)

# Register custom tool
from aitor.tools import Tool
custom_tool = Tool(
    name="search",
    func=search_function,
    description="Search for information"
)
await agent.register_tool(custom_tool)
```

#### Problem Solving
```python
# Solve problems using ReAct reasoning
response = await agent.solve("What is 2+2 and what's the current time?")
print(response)

# Chat interface (alias for solve)
response = await agent.chat("Help me analyze this text")
print(response)
```

### ReAct Reasoning Flow

1. **Think**: Agent analyzes the problem and plans approach
2. **Act**: Agent executes tools to gather information or perform actions
3. **Observe**: Agent analyzes tool results and continues reasoning
4. **Final Answer**: Agent provides complete solution when sufficient information is gathered

### Memory Management

#### Memory Structure
- **Conversation History**: User/assistant messages with timestamps
- **Tool Executions**: Complete tool execution records with results
- **Reasoning Trace**: Step-by-step reasoning with Think/Act/Observe steps
- **Context Management**: Automatic pruning and context window management

#### Memory Operations
```python
# Export/import memory for persistence
memory_data = agent.export_memory()
new_agent.import_memory(memory_data)

# Clear session
agent.clear_session()

# Get memory statistics
stats = agent.get_memory_stats()
```

### LLM Provider Configuration

#### Setup Multiple Providers
```python
from aitor.llm import LLMManager

llm_manager = LLMManager()

# Add OpenAI provider
llm_manager.add_provider(
    name="openai",
    provider="openai",
    config={"api_key": "...", "model": "gpt-4"}
)

# Add Anthropic provider
llm_manager.add_provider(
    name="claude",
    provider="anthropic", 
    config={"api_key": "...", "model": "claude-3-opus-20240229"}
)

# Switch providers
agent.set_llm_provider("claude")
```

### Error Handling

#### Configuration
```python
agent = await (ReactAgentBuilder()
               .max_reasoning_steps(20)  # Limit reasoning steps
               .max_errors(3)           # Maximum errors before stopping
               .build())
```

#### Error Recovery
- Tool execution failures are captured and added to reasoning trace
- Agent can retry with different approaches
- Graceful degradation when error limits are reached

### Example Tools

Common tool patterns included in `examples/example_tools.py`:
- **Mathematical**: Calculator, random number generation
- **Text Processing**: Analysis, transformation
- **File Operations**: Read/write files, directory listing
- **Time/Date**: Current time, timestamp conversion
- **Data Processing**: JSON parsing, list sorting
- **System**: Memory info, sleep operations

### Testing and Development

#### Running Examples
```bash
# Run comprehensive examples
python examples/react_agent_example.py

# Test individual components
python examples/example_tools.py
```

#### Mock LLM for Testing
```python
# Configure mock LLM for testing
llm_manager.add_provider(
    name="mock",
    provider="mock",
    config={
        "responses": {
            "calculate": "ACT: calculator(expression='2+2')",
            "search": "ACT: search(query='test')"
        },
        "default_response": "THINK: I need to analyze this problem."
    }
)
```

### Integration with Base Aitor

ReAct agents seamlessly integrate with the base Aitor framework:
- **Memory Management**: Uses Aitor's thread-safe memory system
- **Async Processing**: Built on Aitor's async execution model
- **Workflow Integration**: Can be used within Aitorflows
- **Thread Safety**: Inherits Aitor's thread-safe design

### Best Practices

1. **Tool Design**: Create focused, single-purpose tools with clear descriptions
2. **Memory Management**: Configure appropriate memory limits for your use case
3. **Error Handling**: Set reasonable error limits and provide fallback strategies
4. **LLM Selection**: Choose appropriate LLM providers based on task complexity
5. **Testing**: Use mock LLM for development and testing

## Current Limitations
- No test framework configured
- No type checking setup (mypy, etc.)
- No linting/formatting tools configured
- Memory persistence is not implemented (export/import available)
- No configuration management system
- Limited error recovery mechanisms in base Aitor (improved in ReAct agents)