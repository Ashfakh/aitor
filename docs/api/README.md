# API Reference

Complete API documentation for the Aitor framework.

## Core Components

### Agents
- [**Aitor**](agents.md#aitor) - Base memory-enabled agent
- [**ReactAgent**](agents.md#reactagent) - Reasoning and acting agent
- [**PlanningReactAgent**](agents.md#planningreactagent) - Planning agent with todos

### Workflows  
- [**Task**](workflows.md#task) - Task wrapper and decorator
- [**Aitorflow**](workflows.md#aitorflow) - DAG workflow execution

### Memory System
- [**ReactMemory**](memory.md#reactmemory) - Structured agent memory
- [**Message**](memory.md#message) - Conversation message
- [**ToolExecution**](memory.md#toolexecution) - Tool execution record

### Tools
- [**Tool**](tools.md#tool) - Base tool class
- [**ToolRegistry**](tools.md#toolregistry) - Tool management
- [**@tool**](tools.md#tool-decorator) - Tool decorator

### LLM Integration
- [**LLMManager**](llm.md#llmmanager) - Multi-provider LLM management
- [**LLMProvider**](llm.md#llmprovider) - Provider interface

### Reasoning
- [**ReasoningEngine**](reasoning.md#reasoningengine) - ReAct reasoning loop
- [**PlanningReasoningEngine**](reasoning.md#planningreasoningengine) - Planning-specific reasoning

### Todo Management
- [**TodoManager**](todo.md#todomanager) - Todo creation and tracking
- [**Todo**](todo.md#todo) - Individual todo item

## Quick Reference

### Common Imports
```python
# Core agents
from aitor import Aitor, create_react_agent, PlanningReactAgent

# Workflows
from aitor import Aitorflow, task

# Tools
from aitor.tools import tool, Tool, ToolRegistry

# Memory
from aitor.memory import ReactMemory, Message

# LLM
from aitor.llm import LLMManager

# Built-in tools
from tools import google_web_search, slack_send_message
```

### Factory Functions
```python
# Create ReAct agent with minimal configuration
agent = await create_react_agent(
    name="MyAgent",
    llm_provider="openai", 
    llm_config={"api_key": "...", "model": "gpt-4"},
    tools=[tool1, tool2]
)
```

### Common Patterns
```python
# Task chaining
task_a >> task_b >> task_c

# Memory management
memory = agent.get_memory()
agent.set_memory(updated_memory)

# Tool registration
@tool(name="example", description="Example tool")
def example_tool(param: str) -> str:
    return f"Result: {param}"

await agent.register_tool(example_tool)
```