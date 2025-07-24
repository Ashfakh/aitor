# Agents API Reference

## Aitor

Base memory-enabled agent class with generic typed memory management.

### Class Definition
```python
class Aitor[T]:
    def __init__(
        self,
        initial_memory: T,
        name: str,
        on_receive_handler: Optional[Callable] = None
    )
```

### Parameters
- **`initial_memory: T`** - Initial memory object of type T
- **`name: str`** - Agent name for identification
- **`on_receive_handler: Optional[Callable]`** - Custom message handler function

### Methods

#### `get_memory() -> T`
Get current memory state (thread-safe).

**Returns:** Current memory object of type T

#### `set_memory(memory: T) -> None`
Update memory state (thread-safe).

**Parameters:**
- **`memory: T`** - New memory object

#### `attach_workflow(workflow: Aitorflow) -> None`
Attach a workflow for execution.

**Parameters:**
- **`workflow: Aitorflow`** - Workflow to attach

#### `async ask(message: str) -> Any`
Send message and wait for response (blocking).

**Parameters:**
- **`message: str`** - Input message

**Returns:** Response from handler or workflow

#### `tell(message: str) -> None`
Send message without waiting for response (non-blocking).

**Parameters:**
- **`message: str`** - Input message

#### `@classmethod shutdown() -> None`
Shutdown class-level thread pool and cleanup resources.

### Example Usage
```python
from typing import List
from aitor import Aitor, Aitorflow, task

# Create agent with typed memory
aitor = Aitor[List[str]](
    initial_memory=[],
    name="TextProcessor"
)

# Use with workflow
@task
def process_text(text: str) -> str:
    return text.upper()

workflow = Aitorflow()
workflow.add_task(process_text)
aitor.attach_workflow(workflow)

# Execute
result = await aitor.ask("hello world")
print(result)  # {'process_text': 'HELLO WORLD'}

# Cleanup
Aitor.shutdown()
```

---

## ReactAgent

Advanced reasoning agent that extends Aitor with ReAct (Reasoning and Acting) capabilities.

### Class Definition
```python
class ReactAgent(Aitor[ReactMemory]):
    def __init__(
        self,
        name: str,
        llm_manager: LLMManager,
        tool_registry: Optional[ToolRegistry] = None,
        max_reasoning_steps: int = 10,
        max_errors: int = 3
    )
```

### Parameters
- **`name: str`** - Agent name
- **`llm_manager: LLMManager`** - LLM provider manager
- **`tool_registry: Optional[ToolRegistry]`** - Tool registry (creates new if None)
- **`max_reasoning_steps: int`** - Maximum reasoning iterations (default: 10)
- **`max_errors: int`** - Maximum errors before stopping (default: 3)

### Methods

#### `async register_tool(tool: Tool) -> None`
Register a tool for agent use.

**Parameters:**
- **`tool: Tool`** - Tool instance to register

#### `async solve(problem: str) -> str`
Solve a problem using ReAct reasoning.

**Parameters:**
- **`problem: str`** - Problem description

**Returns:** Solution or final answer

#### `async chat(message: str) -> str`
Chat interface (alias for solve).

**Parameters:**
- **`message: str`** - User message

**Returns:** Agent response

#### `export_memory() -> Dict[str, Any]`
Export memory for persistence.

**Returns:** Serializable memory data

#### `import_memory(memory_data: Dict[str, Any]) -> None`
Import memory from exported data.

**Parameters:**
- **`memory_data: Dict[str, Any]`** - Previously exported memory

#### `clear_session() -> None`
Clear conversation history and reset session.

#### `get_memory_stats() -> Dict[str, int]`
Get memory usage statistics.

**Returns:** Dictionary with memory metrics

### Example Usage
```python
from aitor import create_react_agent
from aitor.tools import tool

@tool(name="calculator", description="Perform calculations")
def calculate(expression: str) -> float:
    return eval(expression)

# Create agent
agent = await create_react_agent(
    name="MathAgent",
    llm_provider="openai",
    llm_config={"api_key": "your-key", "model": "gpt-4"},
    tools=[calculate],
    max_reasoning_steps=15
)

# Solve problem
response = await agent.solve(
    "What is the square root of 144 plus 10?"
)
print(response)

# Export memory for persistence
memory_data = agent.export_memory()

# Cleanup
await agent.shutdown()
```

---

## PlanningReactAgent

Advanced planning agent that extends ReactAgent with todo management and sub-agent capabilities.

### Class Definition
```python
class PlanningReactAgent(ReactAgent):
    def __init__(
        self,
        name: str,
        llm_manager: LLMManager,
        tool_registry: Optional[ToolRegistry] = None,
        max_reasoning_steps: int = 20,
        max_errors: int = 3,
        max_todos: int = 50
    )
```

### Parameters
- **`name: str`** - Agent name
- **`llm_manager: LLMManager`** - LLM provider manager
- **`tool_registry: Optional[ToolRegistry]`** - Tool registry
- **`max_reasoning_steps: int`** - Maximum reasoning iterations (default: 20)
- **`max_errors: int`** - Maximum errors before stopping (default: 3)
- **`max_todos: int`** - Maximum todos that can be created (default: 50)

### Methods

#### `get_todos() -> List[Todo]`
Get all todos created by the agent.

**Returns:** List of Todo objects

#### `get_pending_todos() -> List[Todo]`
Get todos that are not yet completed.

**Returns:** List of pending Todo objects

#### `get_completed_todos() -> List[Todo]`
Get todos that have been completed.

**Returns:** List of completed Todo objects

#### `create_sub_agent(specialization: str, tools: List[Tool]) -> str`
Create a specialized sub-agent.

**Parameters:**
- **`specialization: str`** - Sub-agent specialization description
- **`tools: List[Tool]`** - Tools for the sub-agent

**Returns:** Sub-agent ID

#### `delegate_to_sub_agent(sub_agent_id: str, task: str) -> str`
Delegate a task to a sub-agent.

**Parameters:**
- **`sub_agent_id: str`** - ID of sub-agent
- **`task: str`** - Task description

**Returns:** Sub-agent response

### Example Usage
```python
from aitor import PlanningReactAgent
from aitor.llm import LLMManager

# Setup LLM
llm_manager = LLMManager()
llm_manager.add_provider("openai", "openai", {
    "api_key": "your-key", 
    "model": "gpt-4"
})

# Create planning agent
agent = PlanningReactAgent(
    name="ProjectPlanner",
    llm_manager=llm_manager,
    max_todos=100
)

# Complex planning task
response = await agent.solve(
    "Plan and execute a data analysis project. Include data collection, "
    "cleaning, analysis, and visualization steps."
)

# Check created todos
todos = agent.get_todos()
print(f"Created {len(todos)} todos")

for todo in todos:
    print(f"- [{todo.status}] {todo.title} (Priority: {todo.priority})")

# Check completion status
completed = agent.get_completed_todos()
pending = agent.get_pending_todos()
print(f"Completed: {len(completed)}, Pending: {len(pending)}")

await agent.shutdown()
```

---

## Factory Functions

### create_react_agent

Convenience function to create a ReactAgent with minimal configuration.

```python
async def create_react_agent(
    name: str,
    llm_provider: str,
    llm_config: Dict[str, Any],
    tools: Optional[List[Tool]] = None,
    max_reasoning_steps: int = 10,
    max_errors: int = 3,
    agent_role: Optional[str] = None,
    additional_instructions: Optional[str] = None
) -> ReactAgent
```

### Parameters
- **`name: str`** - Agent name
- **`llm_provider: str`** - LLM provider name ("openai", "anthropic", "mock")
- **`llm_config: Dict[str, Any]`** - Provider-specific configuration
- **`tools: Optional[List[Tool]]`** - Tools to register with agent
- **`max_reasoning_steps: int`** - Maximum reasoning iterations
- **`max_errors: int`** - Maximum errors before stopping
- **`agent_role: Optional[str]`** - Role description for the agent
- **`additional_instructions: Optional[str]`** - Extra instructions

### Returns
- **`ReactAgent`** - Configured ReactAgent instance

### Example
```python
agent = await create_react_agent(
    name="CustomerSupport",
    llm_provider="openai",
    llm_config={
        "api_key": "sk-...",
        "model": "gpt-4",
        "temperature": 0.7
    },
    tools=[search_tool, database_tool],
    agent_role="customer support representative",
    additional_instructions="Always be helpful and professional"
)
```