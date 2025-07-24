# Aitor

Aitor is a Python framework for building intelligent, memory-enabled agents that can execute complex workflows with advanced reasoning capabilities.

## 🚀 Key Features

- **Memory-Enabled Agents**: Stateful agents with typed memory management
- **DAG Workflows**: Task dependency management with intuitive `>>` operator  
- **ReAct Agents**: Reasoning and Acting agents with Think → Act → Observe loops
- **Planning Agents**: Break complex tasks into manageable todos with sub-agent delegation
- **Tool Integration**: Dynamic tool registry with async execution
- **LLM Support**: OpenAI, Anthropic, and custom providers

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
def process_text(text: str) -> str:
    return text.strip().upper()

@task
def count_words(text: str) -> int:
    return len(text.split())

async def main():
    # Create workflow with task dependencies
    workflow = Aitorflow(name="TextProcessor")
    process_text >> count_words  # Define dependency
    workflow.add_task(process_text)
    
    # Create agent with typed memory
    aitor = Aitor[List[str]](initial_memory=[], name="TextAgent")
    aitor.attach_workflow(workflow)
    
    # Execute
    result = await aitor.ask("  hello world  ")
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

@tool(name="calculator", description="Perform calculations")
def calculate(expression: str) -> float:
    return eval(expression)

async def main():
    agent = await create_react_agent(
        name="MathAgent",
        llm_provider="openai",
        llm_config={"api_key": "your-key", "model": "gpt-4"},
        tools=[calculate]
    )
    
    response = await agent.solve("What is the square root of 144?")
    print(f"Response: {response}")
    
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Planning Agent

```python
from aitor import PlanningReactAgent
from aitor.llm import LLMManager

async def main():
    llm_manager = LLMManager()
    llm_manager.add_provider("openai", "openai", {
        "api_key": "your-key", "model": "gpt-4"
    })
    
    agent = PlanningReactAgent(
        name="Planner", 
        llm_manager=llm_manager
    )
    
    response = await agent.solve(
        "Plan a data analysis project with API data collection, "
        "cleaning, analysis, and visualization."
    )
    
    print(f"Plan: {response}")
    await agent.shutdown()
```

## 🧠 Agent Types

| Agent Type | Description | Use Cases |
|-----------|-------------|-----------|
| **Aitor** | Base memory-enabled agent with workflow support | Data pipelines, task automation |
| **ReAct** | Reasoning agents with tool integration | AI assistants, problem solving |
| **Planning** | Complex task breakdown with sub-agents | Project planning, research automation |

## 🛠️ Built-in Tools

- **Web Search**: Google Custom Search and Tavily integration
- **Slack Integration**: Send messages, files, and manage channels  
- **File Operations**: Read, write, and process various file formats
- **Mathematical**: Calculator and statistical functions
- **System**: Process monitoring and system information

## 📚 Documentation

- [**Architecture Guide**](docs/architecture.md) - Framework design and components
- [**API Reference**](docs/api/) - Detailed API documentation
- [**Tools Development**](docs/tools.md) - Creating custom tools
- [**Examples & Tutorials**](docs/examples/) - Step-by-step guides
- [**Memory Management**](docs/memory.md) - Working with agent memory
- [**LLM Integration**](docs/llm.md) - Configuring language models

## 🧪 Development

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -e .

# Testing
uv run ruff check .      # Linting
uv run mypy src/         # Type checking  
uv run pytest           # Tests

# Examples
python examples/react_agent_example.py
python examples/planning_agent_example.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Aitor** - Build intelligent agents that think, plan, and act. 🤖✨