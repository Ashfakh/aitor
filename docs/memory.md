# Memory Management

Aitor's memory system provides sophisticated state management for agents, supporting both simple typed memory and complex structured memory for advanced agents.

## Overview

All Aitor agents are built around memory-centric design:
- **Type Safety**: Generic memory types ensure compile-time safety
- **Thread Safety**: All memory operations are protected by locks
- **Persistence**: Memory can be exported/imported for session management
- **Structured Storage**: Different memory types for different agent capabilities

## Base Memory System

### Generic Typed Memory

The base `Aitor` class uses generic typed memory:

```python
from typing import List, Dict, Set
from aitor import Aitor

# Agent with list memory
list_agent = Aitor[List[str]](
    initial_memory=[],
    name="ListAgent"
)

# Agent with dictionary memory
dict_agent = Aitor[Dict[str, int]](
    initial_memory={},
    name="DictAgent"
)

# Agent with custom type memory
@dataclass
class UserSession:
    user_id: str
    preferences: Dict[str, Any]
    activity_log: List[str]

session_agent = Aitor[UserSession](
    initial_memory=UserSession(
        user_id="user123",
        preferences={},
        activity_log=[]
    ),
    name="SessionAgent"
)
```

### Memory Operations

```python
# Thread-safe memory access
current_memory = agent.get_memory()

# Modify memory
if isinstance(current_memory, list):
    current_memory.append("new_item")
elif isinstance(current_memory, dict):
    current_memory["new_key"] = "new_value"

# Update memory (thread-safe)
agent.set_memory(current_memory)
```

## ReactMemory System

Advanced agents use structured `ReactMemory` for conversation history, tool executions, and reasoning traces.

### ReactMemory Structure

```python
@dataclass
class ReactMemory:
    conversation_history: List[Message]
    tool_executions: List[ToolExecution] 
    reasoning_trace: List[ReasoningStep]
    context: Dict[str, Any]
    todos: List[Todo]  # For planning agents
```

### Message Management

```python
from aitor.memory import ReactMemory, Message

# Access agent memory
memory = agent.get_memory()

# Add conversation messages
memory.add_message("user", "Hello, how are you?")
memory.add_message("assistant", "I'm doing well, thank you!")

# Add system message
memory.add_message("system", "Remember to be helpful and concise")

# Get conversation history
messages = memory.get_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

# Get messages by role
user_messages = memory.get_messages_by_role("user")
assistant_messages = memory.get_messages_by_role("assistant")
```

### Tool Execution History

```python
from aitor.memory import ToolExecution, ToolResult

# Tool executions are automatically recorded
# But you can also add them manually

tool_result = ToolResult(
    success=True,
    result={"calculation": 42},
    execution_time=0.15,
    tool_name="calculator"
)

tool_execution = ToolExecution(
    tool_name="calculator",
    params={"expression": "6 * 7"},
    result=tool_result,
    timestamp=datetime.now()
)

memory.add_tool_execution(tool_execution)

# Query tool history
all_executions = memory.tool_executions
successful_executions = [ex for ex in all_executions if ex.result.success]
failed_executions = [ex for ex in all_executions if not ex.result.success]

print(f"Total executions: {len(all_executions)}")
print(f"Success rate: {len(successful_executions) / len(all_executions) * 100:.1f}%")
```

### Reasoning Trace

```python
from aitor.memory import ReasoningStep

# Reasoning steps are automatically added during ReAct loops
# But you can query them

memory = agent.get_memory()
reasoning_steps = memory.reasoning_trace

# Analyze reasoning patterns
think_steps = [step for step in reasoning_steps if step.step_type == "THINK"]
act_steps = [step for step in reasoning_steps if step.step_type == "ACT"]
observe_steps = [step for step in reasoning_steps if step.step_type == "OBSERVE"]

print(f"Reasoning breakdown:")
print(f"  THINK steps: {len(think_steps)}")
print(f"  ACT steps: {len(act_steps)}")
print(f"  OBSERVE steps: {len(observe_steps)}")

# Get recent reasoning
recent_steps = reasoning_steps[-10:]  # Last 10 steps
for step in recent_steps:
    print(f"{step.step_type}: {step.content[:100]}...")
```

## Memory Persistence

### Export and Import

```python
# Export memory for persistence
memory_data = agent.export_memory()

# Save to file
import json
with open("agent_memory.json", "w") as f:
    json.dump(memory_data, f, indent=2, default=str)

# Load from file
with open("agent_memory.json", "r") as f:
    loaded_memory = json.load(f)

# Create new agent with imported memory
new_agent = await create_react_agent(
    name="RestoredAgent",
    llm_provider="openai",
    llm_config=llm_config
)
new_agent.import_memory(loaded_memory)
```

### Session Management

```python
class SessionManager:
    """Manage agent sessions with persistent memory."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_session(self, agent: ReactAgent, session_id: str):
        """Save agent memory to disk."""
        memory_data = agent.export_memory()
        session_file = self.storage_path / f"{session_id}.json"
        
        with open(session_file, "w") as f:
            json.dump(memory_data, f, indent=2, default=str)
    
    def load_session(self, agent: ReactAgent, session_id: str) -> bool:
        """Load agent memory from disk."""
        session_file = self.storage_path / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        with open(session_file, "r") as f:
            memory_data = json.load(f)
        
        agent.import_memory(memory_data)
        return True
    
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        return [f.stem for f in self.storage_path.glob("*.json")]

# Usage
session_manager = SessionManager("./agent_sessions")

# Save current session
session_manager.save_session(agent, "user123_session1")

# Load previous session
if session_manager.load_session(agent, "user123_session1"):
    print("Session restored successfully")
else:
    print("No previous session found")
```

## Memory Optimization

### Memory Pruning

Prevent memory from growing too large by implementing automatic pruning:

```python
class ManagedReactAgent(ReactAgent):
    """ReactAgent with automatic memory management."""
    
    def __init__(self, max_messages: int = 100, max_tool_executions: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.max_messages = max_messages
        self.max_tool_executions = max_tool_executions
    
    def prune_memory(self):
        """Remove old entries to keep memory size manageable."""
        memory = self.get_memory()
        
        # Keep recent messages, but preserve system messages
        if len(memory.conversation_history) > self.max_messages:
            system_messages = [msg for msg in memory.conversation_history if msg.role == "system"]
            recent_messages = memory.conversation_history[-self.max_messages:]
            
            # Combine system messages with recent messages
            memory.conversation_history = system_messages + recent_messages
            
            # Remove duplicates while preserving order
            seen = set()
            unique_messages = []
            for msg in memory.conversation_history:
                msg_key = (msg.role, msg.content)
                if msg_key not in seen:
                    seen.add(msg_key)
                    unique_messages.append(msg)
            
            memory.conversation_history = unique_messages
        
        # Keep recent tool executions
        if len(memory.tool_executions) > self.max_tool_executions:
            memory.tool_executions = memory.tool_executions[-self.max_tool_executions:]
        
        # Keep reasoning trace manageable
        if len(memory.reasoning_trace) > 200:
            memory.reasoning_trace = memory.reasoning_trace[-200:]
        
        self.set_memory(memory)
    
    async def solve(self, problem: str) -> str:
        """Solve problem with automatic memory pruning."""
        # Prune memory before processing
        self.prune_memory()
        
        # Process normally
        result = await super().solve(problem)
        
        # Prune memory after processing if needed
        memory = self.get_memory()
        if (len(memory.conversation_history) > self.max_messages * 1.2 or
            len(memory.tool_executions) > self.max_tool_executions * 1.2):
            self.prune_memory()
        
        return result

# Usage
agent = ManagedReactAgent(
    name="ManagedAgent",
    llm_manager=llm_manager,
    max_messages=50,  # Keep last 50 messages
    max_tool_executions=25  # Keep last 25 tool executions
)
```

### Memory Statistics

```python
def analyze_memory_usage(agent: ReactAgent) -> Dict[str, Any]:
    """Analyze agent memory usage and patterns."""
    memory = agent.get_memory()
    
    # Message statistics
    message_stats = {
        "total_messages": len(memory.conversation_history),
        "by_role": {}
    }
    
    for role in ["user", "assistant", "system"]:
        role_messages = memory.get_messages_by_role(role)
        message_stats["by_role"][role] = len(role_messages)
        
        if role_messages:
            avg_length = sum(len(msg.content) for msg in role_messages) / len(role_messages)
            message_stats["by_role"][f"{role}_avg_length"] = round(avg_length, 2)
    
    # Tool execution statistics
    tool_stats = {
        "total_executions": len(memory.tool_executions),
        "success_rate": 0,
        "avg_execution_time": 0,
        "tool_usage": {}
    }
    
    if memory.tool_executions:
        successful = [ex for ex in memory.tool_executions if ex.result.success]
        tool_stats["success_rate"] = len(successful) / len(memory.tool_executions)
        
        total_time = sum(ex.result.execution_time for ex in memory.tool_executions)
        tool_stats["avg_execution_time"] = total_time / len(memory.tool_executions)
        
        # Tool usage frequency
        for execution in memory.tool_executions:
            tool_name = execution.tool_name
            tool_stats["tool_usage"][tool_name] = tool_stats["tool_usage"].get(tool_name, 0) + 1
    
    # Reasoning statistics
    reasoning_stats = {
        "total_steps": len(memory.reasoning_trace),
        "step_distribution": {}
    }
    
    for step in memory.reasoning_trace:
        step_type = step.step_type
        reasoning_stats["step_distribution"][step_type] = (
            reasoning_stats["step_distribution"].get(step_type, 0) + 1
        )
    
    # Memory size estimation (rough)
    memory_size = {
        "messages_kb": sum(len(msg.content.encode()) for msg in memory.conversation_history) / 1024,
        "tools_kb": len(str(memory.tool_executions).encode()) / 1024,
        "reasoning_kb": sum(len(step.content.encode()) for step in memory.reasoning_trace) / 1024
    }
    memory_size["total_kb"] = sum(memory_size.values())
    
    return {
        "messages": message_stats,
        "tools": tool_stats,
        "reasoning": reasoning_stats,
        "memory_size": memory_size,
        "analysis_timestamp": datetime.now().isoformat()
    }

# Usage
stats = analyze_memory_usage(agent)
print(f"Memory Analysis:")
print(f"  Total Messages: {stats['messages']['total_messages']}")
print(f"  Tool Success Rate: {stats['tools']['success_rate']:.2%}")
print(f"  Memory Size: {stats['memory_size']['total_kb']:.1f} KB")
print(f"  Most Used Tool: {max(stats['tools']['tool_usage'].items(), key=lambda x: x[1]) if stats['tools']['tool_usage'] else 'None'}")
```

## Custom Memory Types

### Domain-Specific Memory

Create custom memory structures for specialized applications:

```python
@dataclass
class CustomerSupportMemory:
    """Specialized memory for customer support agents."""
    conversation_history: List[Message]
    tool_executions: List[ToolExecution]
    reasoning_trace: List[ReasoningStep]
    context: Dict[str, Any]
    
    # Customer support specific fields
    customer_id: Optional[str] = None
    ticket_id: Optional[str] = None
    issue_category: Optional[str] = None
    resolution_status: str = "open"
    escalation_level: int = 1
    customer_satisfaction: Optional[int] = None
    interaction_notes: List[str] = field(default_factory=list)
    
    def add_interaction_note(self, note: str):
        """Add a note about the customer interaction."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.interaction_notes.append(f"[{timestamp}] {note}")
    
    def escalate(self, reason: str):
        """Escalate the support case."""
        self.escalation_level += 1
        self.add_interaction_note(f"Escalated to level {self.escalation_level}: {reason}")
    
    def resolve(self, resolution: str, satisfaction: int):
        """Mark the case as resolved."""
        self.resolution_status = "resolved"
        self.customer_satisfaction = satisfaction
        self.add_interaction_note(f"Resolved: {resolution} (Satisfaction: {satisfaction}/5)")

class CustomerSupportAgent(ReactAgent):
    """Specialized agent for customer support with custom memory."""
    
    def __init__(self, **kwargs):
        # Initialize with custom memory
        custom_memory = CustomerSupportMemory(
            conversation_history=[],
            tool_executions=[],
            reasoning_trace=[],
            context={}
        )
        
        super().__init__(**kwargs)
        self.set_memory(custom_memory)
    
    def start_support_session(self, customer_id: str, issue_category: str):
        """Start a new support session."""
        memory = self.get_memory()
        memory.customer_id = customer_id
        memory.issue_category = issue_category
        memory.ticket_id = f"TICKET_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{customer_id}"
        memory.add_interaction_note(f"Support session started for {customer_id}, category: {issue_category}")
        self.set_memory(memory)
    
    def get_support_status(self) -> Dict[str, Any]:
        """Get current support session status."""
        memory = self.get_memory()
        return {
            "customer_id": memory.customer_id,
            "ticket_id": memory.ticket_id,
            "issue_category": memory.issue_category,
            "resolution_status": memory.resolution_status,
            "escalation_level": memory.escalation_level,
            "interaction_count": len(memory.interaction_notes),
            "customer_satisfaction": memory.customer_satisfaction
        }
```

### Memory Hooks

Implement memory hooks for automatic processing:

```python
class HookedReactAgent(ReactAgent):
    """Agent with memory hooks for automatic processing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_hooks = []
    
    def add_memory_hook(self, hook_func):
        """Add a function to be called whenever memory is updated."""
        self.memory_hooks.append(hook_func)
    
    def set_memory(self, memory):
        """Override to call hooks after memory update."""
        super().set_memory(memory)
        
        # Call all hooks
        for hook in self.memory_hooks:
            try:
                hook(self, memory)
            except Exception as e:
                print(f"Memory hook error: {e}")

# Example hooks
def log_memory_changes(agent, memory):
    """Log memory changes to file."""
    with open("memory_changes.log", "a") as f:
        f.write(f"{datetime.now()}: Agent {agent.name} memory updated\n")

def detect_long_conversations(agent, memory):
    """Alert when conversations get too long."""
    if len(memory.conversation_history) > 100:
        print(f"Warning: Agent {agent.name} has {len(memory.conversation_history)} messages")

def auto_summarize_old_messages(agent, memory):
    """Automatically summarize old messages to save space."""
    if len(memory.conversation_history) > 200:
        # Implement summarization logic here
        print(f"Auto-summarizing old messages for agent {agent.name}")

# Usage
agent = HookedReactAgent(name="HookedAgent", llm_manager=llm_manager)
agent.add_memory_hook(log_memory_changes)
agent.add_memory_hook(detect_long_conversations)
agent.add_memory_hook(auto_summarize_old_messages)
```

## Best Practices

### 1. Memory Size Management
- Implement automatic pruning for long-running agents
- Monitor memory usage regularly
- Use appropriate data structures for your use case

### 2. Thread Safety
- Always use `get_memory()` and `set_memory()` for memory access
- Never modify memory directly without proper synchronization
- Be aware that memory operations are atomic but sequences of operations are not

### 3. Persistence Strategy
- Export memory regularly for important agents
- Use structured storage for complex memory types
- Implement backup and recovery procedures

### 4. Performance Optimization
- Use appropriate memory types for your data
- Consider memory pools for high-frequency operations
- Monitor memory growth patterns

### 5. Security Considerations
- Don't store sensitive data in memory unless necessary
- Implement memory encryption for sensitive applications
- Clear memory appropriately when sessions end

The memory system is the foundation of Aitor's stateful capabilities. Understanding how to effectively manage memory will help you build more sophisticated and reliable agents.