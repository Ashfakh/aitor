# Aitor

Aitor is a comprehensive Python framework for building intelligent, memory-enabled agents that can execute complex workflows with advanced reasoning capabilities. The framework combines DAG-based workflow execution with ReAct (Reasoning and Acting) agents, planning agents, and LLM integration to create powerful AI-driven applications.

## 🚀 Features

### Core Framework
- **Memory-Enabled Agents**: Stateful agents with typed memory management and persistence
- **DAG Workflows**: Task dependency management through directed acyclic graphs
- **Async Processing**: Thread-safe execution with both blocking (`ask`) and non-blocking (`tell`) APIs
- **Task Chaining**: Intuitive `>>` operator for defining task dependencies

### AI Agent Types
- **ReAct Agents**: Reasoning and Acting agents with Think → Act → Observe loops
- **Planning Agents**: Advanced agents that break down complex tasks into manageable todos
- **Sub-Agent Management**: Delegate specialized tasks to focused sub-agents
- **LLM Integration**: Support for OpenAI, Anthropic, and custom LLM providers

### Advanced Capabilities
- **Tool Registry**: Dynamic tool management with async execution
- **Structured Responses**: JSON-based responses using Pydantic models
- **Memory Persistence**: Export/import memory for session management
- **Workflow Visualization**: Built-in Mermaid diagram generation

## 📦 Installation

```bash
pip install aitor
```

For development:
```bash
git clone https://github.com/Ashfakh/aitor.git
cd aitor
pip install -e .
```

## 🏃 Quick Start

### Basic Workflow Agent

```python
import asyncio
from typing import List
from aitor import Aitor, Aitorflow, task

@task
def clean_text(text: str) -> str:
    return text.strip().lower()

@task
def count_words(text: str) -> int:
    return len(text.split())

@task
def analyze_text(text: str, word_count: int) -> dict:
    return {
        "text": text,
        "word_count": word_count,
        "avg_word_length": len(text.replace(" ", "")) / word_count if word_count > 0 else 0
    }

async def text_handler(message: str, aitor: Aitor[List[str]]):
    # Store in memory
    memory = aitor.get_memory()
    memory.append(message)
    aitor.set_memory(memory)
    
    # Execute workflow if attached
    if aitor.workflow:
        return await asyncio.to_thread(aitor.workflow.execute, message)
    return f"Processed: {message}"

async def main():
    # Create workflow
    workflow = Aitorflow(name="Text Analysis")
    
    # Define task dependencies
    clean_text >> count_words >> analyze_text
    clean_text >> analyze_text  # Multiple dependencies
    
    workflow.add_task(clean_text)
    
    # Create agent
    aitor = Aitor(
        initial_memory=[],
        name="TextProcessor",
        on_receive_handler=text_handler
    )
    aitor.attach_workflow(workflow)
    
    # Process text
    result = await aitor.ask("  Hello World Example!  ")
    print(f"Result: {result}")
    
    Aitor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### ReAct Agent with Tools

```python
import asyncio
from aitor import create_react_agent
from aitor.tools import tool

@tool(name="calculator", description="Perform mathematical calculations")
def calculate(expression: str) -> float:
    """Safely evaluate mathematical expressions."""
    try:
        # Simple validation for safety
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        return eval(expression)
    except Exception as e:
        raise ValueError(f"Calculation error: {e}")

@tool(name="text_analyzer", description="Analyze text properties")
def analyze_text(text: str) -> dict:
    """Analyze various properties of text."""
    return {
        "length": len(text),
        "word_count": len(text.split()),
        "sentence_count": text.count('.') + text.count('!') + text.count('?'),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }

async def main():
    # Create ReAct agent
    agent = await create_react_agent(
        name="MathTextAgent",
        llm_provider="openai",
        llm_config={
            "api_key": "your-openai-api-key",
            "model": "gpt-4"
        },
        tools=[calculate, analyze_text],
        max_reasoning_steps=10
    )
    
    # Solve complex problems
    response = await agent.solve(
        "Calculate the square root of 144, then analyze the text 'Hello World!' "
        "and tell me the relationship between the calculation result and word count."
    )
    print(f"Agent Response: {response}")
    
    # Export memory for persistence
    memory_data = agent.export_memory()
    print(f"Conversation history: {len(memory_data['conversation_history'])} messages")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Planning Agent with Todos

```python
import asyncio
from aitor import PlanningReactAgent
from aitor.llm import LLMManager

async def main():
    # Setup LLM
    llm_manager = LLMManager()
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={"api_key": "your-api-key", "model": "gpt-4"}
    )
    
    # Create planning agent
    agent = PlanningReactAgent(
        name="ProjectPlanner",
        llm_manager=llm_manager,
        max_reasoning_steps=20
    )
    
    # Complex planning task
    response = await agent.solve(
        "Help me plan and execute a data analysis project. I need to collect data "
        "from APIs, clean it, perform statistical analysis, and create visualizations."
    )
    
    print(f"Planning Response: {response}")
    
    # Check created todos
    memory = agent.get_memory()
    print(f"Created {len(memory.todos)} todos:")
    for todo in memory.todos:
        print(f"  - [{todo.status}] {todo.title}")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧠 Agent Types

### 1. Base Aitor Agent
Memory-enabled agents with workflow integration:
- Generic typed memory: `Aitor[T]`
- Thread-safe operations
- Workflow attachment
- Async processing

### 2. ReAct Agents
Reasoning and Acting agents that follow Think → Act → Observe loops:
- **Tool Integration**: Dynamic tool registry with validation
- **Reasoning Engine**: Step-by-step problem solving
- **Memory Management**: Conversation and reasoning history
- **LLM Integration**: Support for multiple providers

### 3. Planning Agents
Advanced agents that break complex tasks into manageable todos:
- **Todo Management**: Create, track, and execute todos with priorities
- **Sub-Agent Delegation**: Spawn specialized agents for specific tasks
- **Plan Execution**: Systematic approach to complex problems
- **Progress Tracking**: Monitor todo completion and overall progress

## 🛠️ Core Components

### Memory System
```python
from aitor.memory import ReactMemory

# Structured memory with conversation, tools, and reasoning
memory = ReactMemory()
memory.add_message("user", "Hello!")
memory.add_tool_execution("calculator", {"expression": "2+2"}, 4)
memory.add_reasoning_step("THINK", "I need to solve this math problem")
```

### Tool Registry
```python
from aitor.tools import ToolRegistry, tool

registry = ToolRegistry()

@tool(name="example", description="An example tool")
def example_tool(param: str) -> str:
    return f"Processed: {param}"

await registry.register_tool(example_tool)
result = await registry.execute_tool("example", {"param": "test"})
```

### LLM Management
```python
from aitor.llm import LLMManager

llm_manager = LLMManager()

# Add multiple providers
llm_manager.add_provider("openai", "openai", {
    "api_key": "...", 
    "model": "gpt-4"
})
llm_manager.add_provider("claude", "anthropic", {
    "api_key": "...", 
    "model": "claude-3-opus-20240229"
})

# Switch between providers
llm_manager.set_default_provider("claude")
```

### Workflow Visualization
```python
from aitor import Aitorflow

workflow = Aitorflow(name="Example")
# ... add tasks ...

# Generate Mermaid diagram
mermaid_code = workflow.visualize()
print(mermaid_code)
```

## 📚 Advanced Usage

### Custom Tool Development
```python
from aitor.tools import Tool
import asyncio

class DatabaseTool(Tool):
    def __init__(self, db_connection):
        super().__init__(
            name="database",
            func=self.query_database,
            description="Execute database queries"
        )
        self.db = db_connection
    
    async def query_database(self, query: str) -> list:
        # Async database operations
        return await self.db.execute(query)
```

### Memory Persistence
```python
# Export memory
memory_data = agent.export_memory()

# Save to file
import json
with open("agent_memory.json", "w") as f:
    json.dump(memory_data, f, indent=2)

# Load and import
with open("agent_memory.json", "r") as f:
    memory_data = json.load(f)

new_agent.import_memory(memory_data)
```

### Sub-Agent Architecture
```python
from aitor.sub_agent import SubAgentManager

# Create specialized sub-agents
sub_manager = SubAgentManager()
sub_manager.create_sub_agent(
    name="DataAnalyst",
    specialization="statistical analysis and data processing",
    tools=[pandas_tool, numpy_tool, matplotlib_tool]
)

# Delegate tasks
result = await sub_manager.delegate_task(
    "DataAnalyst",
    "Analyze this dataset and create visualizations"
)
```

## 🎯 Use Cases

- **Data Processing Pipelines**: Complex ETL workflows with error handling
- **AI-Powered Assistants**: Conversational agents with tool access
- **Automated Planning**: Break down complex projects into actionable steps
- **Research Automation**: Gather, analyze, and synthesize information
- **Code Generation**: AI agents that write and test code
- **Content Creation**: Multi-step content workflows with reviews

## 🧪 Development

### Setup Development Environment
```bash
# Clone repository
git clone https://github.com/Ashfakh/aitor.git
cd aitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Running Tests
```bash
# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/

# Run tests (when available)
uv run pytest
```

### Examples
```bash
# Basic workflow example
python example.py

# ReAct agent example
python examples/react_agent_example.py

# Planning agent example
python examples/planning_agent_example.py

# Sales chat agent
python examples/sales_chat_agent.py
```

## 📖 Documentation

For detailed documentation, see:
- **Architecture**: Understanding the framework design
- **Agent Types**: Comprehensive guide to different agent capabilities
- **Tool Development**: Creating custom tools and integrations
- **Memory Management**: Working with agent memory and persistence
- **LLM Integration**: Configuring and using different language models

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Use type hints consistently
- Follow existing code patterns
- Add docstrings to public APIs
- Ensure thread safety for shared resources
- Test your changes thoroughly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT model integration
- Anthropic for Claude model support
- The Python async/await ecosystem
- Contributors and community feedback

---

**Aitor** - Build intelligent agents that think, plan, and act. 🤖✨