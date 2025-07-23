"""
Aitor - A Python framework for building intelligent, memory-enabled agents

Features:
- Memory-enabled agents with typed memory management
- ReAct reasoning framework (Think -> Act -> Observe)
- Planning-based agents with todo management and sub-agents
- Tool execution and management system
- Multi-LLM provider support
- Async workflow processing
"""

__version__ = "0.2.0"

# Core Aitor components
from .aitor import Aitor
from .aitorflows import Aitorflow
from .task import Task, task

# ReAct agent components
from .react_agent import ReactAgent, ReactAgentBuilder, create_react_agent
from .tools import Tool, ToolRegistry, ToolResult, ToolExecution, tool, default_registry
from .memory import ReactMemory, Message, ReasoningStep, MessageRole, ReasoningStepType
from .reasoning import ReasoningEngine
from .llm import (
    LLMManager,
    LLMFactory,
    LLMProvider,
    BaseLLM,
    LLMResponse,
    Message as LLMMessage,
)

# Planning agent components
from .planning_agent import (
    PlanningReactAgent,
    PlanningReactAgentBuilder,
    create_planning_agent,
)
from .planning_reasoning import PlanningReasoningEngine
from .todo import TodoManager, TodoItem, TodoStatus, TodoPriority

# Logging utilities
from .logging_config import (
    setup_aitor_logging,
    enable_aitor_logging,
    disable_aitor_logging,
    is_aitor_logging_enabled,
    set_aitor_logging,
)

__all__ = [
    # Core Aitor
    "Aitor",
    "Aitorflow",
    "Task",
    "task",
    # ReAct Agent
    "ReactAgent",
    "ReactAgentBuilder",
    "create_react_agent",
    # Planning Agent
    "PlanningReactAgent",
    "PlanningReactAgentBuilder",
    "create_planning_agent",
    "PlanningReasoningEngine",
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolExecution",
    "tool",
    "default_registry",
    # Memory
    "ReactMemory",
    "Message",
    "ReasoningStep",
    "MessageRole",
    "ReasoningStepType",
    # Reasoning
    "ReasoningEngine",
    # Todo Management
    "TodoManager",
    "TodoItem",
    "TodoStatus",
    "TodoPriority",
    # LLM
    "LLMManager",
    "LLMFactory",
    "LLMProvider",
    "BaseLLM",
    "LLMResponse",
    "LLMMessage",
    # Logging
    "setup_aitor_logging",
    "enable_aitor_logging",
    "disable_aitor_logging",
    "is_aitor_logging_enabled",
    "set_aitor_logging",
]
