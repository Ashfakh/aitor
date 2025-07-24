# Aitor Framework Architecture

Aitor is designed as a comprehensive Python framework for building intelligent, memory-enabled agents with advanced reasoning capabilities. This document provides an in-depth look at the framework's architecture, design principles, and core components.

## Design Principles

### 1. Memory-Driven Architecture
All agents in Aitor are built around typed memory management, ensuring thread-safe access to persistent state across async operations.

### 2. Compositional Design  
The framework uses composition over inheritance, allowing flexible combination of workflows, tools, and reasoning capabilities.

### 3. Async-First Approach
Built on Python's async/await paradigm with proper thread pool integration for blocking operations.

### 4. Type Safety
Extensive use of generics and type hints ensures compile-time safety and better developer experience.

## Core Architecture Components

### 1. Base Aitor Agent (`src/aitor/aitor.py`)

The foundation agent class that provides:

```python
class Aitor[T]:
    """Base memory-enabled agent with generic typed memory."""
```

**Key Features:**
- **Generic Memory**: `Aitor[List[str]]`, `Aitor[Dict[str, Any]]`
- **Thread Safety**: All memory operations protected by `threading.Lock`
- **Lifecycle Management**: Class-level `ThreadPoolExecutor` shared across instances
- **Message Handling**: Configurable `on_receive_handler` for custom logic

**Threading Model:**
- Class-level thread pool shared across all instances
- Per-thread event loops using `threading.local()`
- Thread-safe memory access with automatic lock management
- Proper cleanup in `shutdown()` method

### 2. Task System (`src/aitor/task.py`)

Lightweight wrapper for creating workflow-compatible functions:

```python
@task
def process_data(input: str) -> dict:
    return {"processed": input.upper()}
```

**Features:**
- Decorator-based task creation
- Intuitive `>>` operator for dependencies: `task_a >> task_b`
- Automatic parameter validation
- Support for both sync and async functions

### 3. Workflow Engine (`src/aitor/aitorflows.py`)

DAG-based workflow execution with parallel processing:

```python
class Aitorflow:
    """Directed Acyclic Graph workflow executor."""
```

**Capabilities:**
- Topological sorting for execution order
- Cycle detection and validation
- Parallel execution of independent tasks
- Mermaid diagram generation for visualization
- Error propagation and workflow termination

**Execution Flow:**
1. **Validation**: Check for cycles and parameter compatibility
2. **Entry Point Detection**: Find tasks with no dependencies
3. **Parallel Execution**: Execute independent tasks concurrently
4. **Result Aggregation**: Collect and pass results to downstream tasks
5. **Exit Point Collection**: Gather final outputs

### 4. ReAct Agent System

#### ReAct Agent (`src/aitor/react_agent.py`)
Advanced reasoning agents that extend the base Aitor with intelligent capabilities:

```python
class ReactAgent(Aitor[ReactMemory]):
    """ReAct agent with reasoning and tool execution."""
```

**Architecture:**
- Extends `Aitor` with `ReactMemory` for structured conversation history
- Integrates `ReasoningEngine` for Think → Act → Observe loops
- Uses `ToolRegistry` for dynamic tool management
- Supports multiple LLM providers through `LLMManager`

#### Reasoning Engine (`src/aitor/reasoning.py`)
Core reasoning loop implementation:

```python
class ReasoningEngine:
    """Orchestrates ReAct reasoning cycles."""
```

**Process:**
1. **THINK**: Agent analyzes the problem and plans approach
2. **ACT**: Agent executes tools to gather information
3. **OBSERVE**: Agent analyzes results and continues reasoning
4. **FINAL ANSWER**: Agent provides solution when complete

**Features:**
- Step limit enforcement
- Error recovery and retry logic
- Structured output parsing using Pydantic models
- Reasoning trace preservation

### 5. Planning Agent System

#### Planning Agent (`src/aitor/planning_agent.py`)
Advanced agents that break down complex tasks:

```python
class PlanningReactAgent(ReactAgent):
    """Planning agent with todo management and sub-agents."""
```

**Capabilities:**
- Todo creation and management with priorities
- Sub-agent delegation for specialized tasks
- Plan → Execute → Adapt reasoning loops
- Progress tracking and reporting

#### Todo Management (`src/aitor/todo.py`)
Task planning and tracking system:

```python
class TodoManager:
    """Manages todos with priorities and status tracking."""
```

**Features:**
- Priority-based todo organization
- Status tracking (pending, in_progress, completed)
- Dependency management between todos
- Progress reporting and statistics

### 6. Memory System (`src/aitor/memory.py`)

Structured memory for different agent types:

#### ReactMemory
```python
@dataclass
class ReactMemory:
    conversation_history: List[Message]
    tool_executions: List[ToolExecution]
    reasoning_trace: List[ReasoningStep]
    context: Dict[str, Any]
```

**Features:**
- Role-based message storage (user, assistant, system)
- Complete tool execution history with results
- Step-by-step reasoning preservation
- Context management with automatic pruning
- JSON serialization for persistence

### 7. Tool System (`src/aitor/tools.py`)

Dynamic tool management with async execution:

```python
class ToolRegistry:
    """Thread-safe registry for agent tools."""
```

**Architecture:**
- Async tool execution with timeout handling
- Parameter validation using function signatures
- JSON schema generation for LLM integration
- Thread-safe registration and execution
- Comprehensive error handling and logging

**Tool Creation Patterns:**
```python
# Decorator approach
@tool(name="calculator", description="Perform calculations")
def calculate(expression: str) -> float:
    return eval(expression)

# Factory pattern for tenant-specific tools
def create_database_tool(tenant_id: str):
    @tool(name=f"db_{tenant_id}")
    def query_db(query: str) -> dict:
        return execute_query(tenant_id, query)
    return query_db
```

### 8. LLM Integration (`src/aitor/llm.py`)

Unified interface for multiple LLM providers:

```python
class LLMManager:
    """Manages multiple LLM providers with failover."""
```

**Supported Providers:**
- OpenAI (GPT-3.5, GPT-4, o3-mini)
- Anthropic (Claude 3.x series)
- Mock provider for testing

**Features:**
- Connection pooling and retry logic
- Provider-specific parameter handling
- Structured JSON response parsing
- Automatic failover between providers
- Usage tracking and monitoring

## Data Flow Architecture

### 1. Basic Aitor Workflow
```
User Input → on_receive_handler → Workflow Execution → Memory Update → Response
```

### 2. ReAct Agent Flow
```
User Query → Reasoning Engine → [Think → Act → Observe]* → Final Answer
                    ↓
              Tool Execution → Memory Update
                    ↓
              LLM Integration → Response Generation
```

### 3. Planning Agent Flow
```
Complex Task → Task Analysis → Todo Creation → Sub-Agent Delegation
                    ↓
              Progress Tracking → Plan Adaptation → Final Completion
```

## Concurrency and Threading

### Thread Safety Guarantees
- All memory operations are protected by locks
- Tool registry uses async locks for thread safety
- Event loops are isolated per thread using `threading.local()`
- Proper cleanup ensures no resource leaks

### Async Execution Model
- Blocking operations run in thread pool via `asyncio.to_thread()`
- Tool execution is async by default with timeout handling
- LLM calls are async with connection pooling
- Workflow tasks can be mixed sync/async

### Resource Management
- Class-level thread pool shared across agent instances
- Automatic cleanup on `shutdown()`
- Context managers for resource acquisition
- Proper exception handling and resource release

## Extensibility Points

### 1. Custom Agent Types
Extend base classes to create specialized agents:
```python
class CustomAgent(ReactAgent):
    def __init__(self, specialized_config):
        super().__init__(...)
        self.custom_behavior = specialized_config
```

### 2. Custom Memory Types
Create domain-specific memory structures:
```python
@dataclass
class CustomMemory:
    domain_data: List[DomainObject]
    custom_indices: Dict[str, int]
```

### 3. Custom Tool Categories
Implement specialized tool types:
```python
class DatabaseTool(Tool):
    def __init__(self, connection_string):
        self.connection = create_connection(connection_string)
        super().__init__(...)
```

### 4. Custom LLM Providers
Add new language model integrations:
```python
class CustomLLMProvider(LLMProvider):
    async def generate_response(self, messages, **kwargs):
        # Custom implementation
        pass
```

## Error Handling Strategy

### 1. Graceful Degradation
- Tool failures don't crash agents
- LLM provider failover
- Memory corruption recovery
- Workflow error isolation

### 2. Comprehensive Logging
- Structured logging with context
- Error tracking with stack traces
- Performance monitoring
- Debug information preservation

### 3. Recovery Mechanisms
- Automatic retry with exponential backoff
- State restoration from memory
- Partial execution results preservation
- User notification of failures

## Performance Considerations

### 1. Memory Management
- Lazy loading of large objects
- Automatic memory pruning
- Efficient serialization formats
- Garbage collection optimization

### 2. Concurrency Optimization
- Optimal thread pool sizing
- Async/await throughout the stack
- Connection pooling for external services
- Batch processing where applicable

### 3. Scalability Features
- Stateless tool execution
- Horizontal scaling support
- Resource pooling
- Configurable limits and timeouts

## Security Architecture

### 1. Input Validation
- Parameter validation for all tools
- SQL injection prevention
- Command injection protection
- Input sanitization

### 2. Sandboxing
- Tool execution isolation
- Resource limit enforcement
- Safe evaluation contexts
- Access control for sensitive operations

### 3. Credential Management
- Environment variable configuration
- No hardcoded secrets
- Secure credential storage
- API key rotation support

This architecture provides a solid foundation for building sophisticated AI agents while maintaining flexibility, performance, and safety.