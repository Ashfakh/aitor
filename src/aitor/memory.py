"""
Memory system for ReAct agents.
Provides structured memory management for conversations, tool executions, and reasoning traces.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

from .tools import ToolExecution, ToolResult

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of a message in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ReasoningStepType(Enum):
    """Type of reasoning step."""

    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    FINAL_ANSWER = "final_answer"


@dataclass
class Message:
    """A message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReasoningStep:
    """A step in the reasoning process."""

    step_type: ReasoningStepType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    tool_result: Optional[ToolResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.step_type, str):
            self.step_type = ReasoningStepType(self.step_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning step to dictionary."""
        data = {
            "step_type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_params:
            data["tool_params"] = self.tool_params
        if self.tool_result:
            data["tool_result"] = {
                "success": self.tool_result.success,
                "result": self.tool_result.result,
                "error": self.tool_result.error,
                "execution_time": self.tool_result.execution_time,
                "tool_name": self.tool_result.tool_name,
            }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create reasoning step from dictionary."""
        step = cls(
            step_type=ReasoningStepType(data["step_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_name=data.get("tool_name"),
            tool_params=data.get("tool_params"),
            metadata=data.get("metadata", {}),
        )

        if "tool_result" in data:
            tr_data = data["tool_result"]
            step.tool_result = ToolResult(
                success=tr_data["success"],
                result=tr_data["result"],
                error=tr_data.get("error"),
                execution_time=tr_data.get("execution_time", 0.0),
                tool_name=tr_data.get("tool_name", ""),
            )

        return step


@dataclass
class ReactMemory:
    """
    Structured memory for ReAct agents.

    Contains conversation history, tool executions, reasoning traces,
    and context management capabilities.
    """

    # Core memory components
    conversation_history: List[Message] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)

    # Context management
    max_conversation_length: int = 100
    max_tool_executions: int = 50
    max_reasoning_steps: int = 100
    context_window_tokens: int = 4000

    # Metadata
    agent_name: str = "ReactAgent"
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def add_message(
        self,
        role: Union[MessageRole, str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        with self._lock:
            message_role = role if isinstance(role, MessageRole) else MessageRole(role)
            message = Message(
                role=message_role, content=content, metadata=metadata or {}
            )
            self.conversation_history.append(message)
            self.last_updated = datetime.now()

            # Prune if necessary
            if len(self.conversation_history) > self.max_conversation_length:
                self._prune_conversation_history()

    def add_tool_execution(self, execution: ToolExecution) -> None:
        """
        Add a tool execution to history.

        Args:
            execution: Tool execution to add
        """
        with self._lock:
            self.tool_executions.append(execution)
            self.last_updated = datetime.now()

            # Prune if necessary
            if len(self.tool_executions) > self.max_tool_executions:
                self._prune_tool_executions()

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """
        Add a reasoning step to trace.

        Args:
            step: Reasoning step to add
        """
        with self._lock:
            self.reasoning_trace.append(step)
            self.last_updated = datetime.now()

            # Prune if necessary
            if len(self.reasoning_trace) > self.max_reasoning_steps:
                self._prune_reasoning_trace()

    def get_conversation_context(self, last_n: Optional[int] = None) -> str:
        """
        Get conversation context as formatted string.

        Args:
            last_n: Number of recent messages to include

        Returns:
            Formatted conversation context
        """
        with self._lock:
            messages = self.conversation_history
            if last_n is not None:
                messages = messages[-last_n:]

            context_lines = []
            for msg in messages:
                role = msg.role.value.upper()
                context_lines.append(f"{role}: {msg.content}")

            return "\n".join(context_lines)

    def get_tool_execution_context(self, last_n: Optional[int] = None) -> str:
        """
        Get tool execution context as formatted string.

        Args:
            last_n: Number of recent executions to include

        Returns:
            Formatted tool execution context
        """
        with self._lock:
            executions = self.tool_executions
            if last_n is not None:
                executions = executions[-last_n:]

            context_lines = []
            for exec in executions:
                result_summary = (
                    "SUCCESS" if exec.result.success else f"ERROR: {exec.result.error}"
                )
                context_lines.append(
                    f"TOOL: {exec.tool_name}({exec.params}) -> {result_summary}"
                )

            return "\n".join(context_lines)

    def get_reasoning_context(self, last_n: Optional[int] = None) -> str:
        """
        Get reasoning trace context as formatted string.

        Args:
            last_n: Number of recent steps to include

        Returns:
            Formatted reasoning context
        """
        with self._lock:
            steps = self.reasoning_trace
            if last_n is not None:
                steps = steps[-last_n:]

            context_lines = []
            for step in steps:
                step_type = step.step_type.value.upper()
                context_lines.append(f"{step_type}: {step.content}")

                if step.tool_name and step.tool_result:
                    result_summary = (
                        "SUCCESS"
                        if step.tool_result.success
                        else f"ERROR: {step.tool_result.error}"
                    )
                    context_lines.append(
                        f"  TOOL_RESULT: {step.tool_name} -> {result_summary}"
                    )

            return "\n".join(context_lines)

    def get_full_context(self) -> str:
        """
        Get complete context including conversation, tools, and reasoning.

        Returns:
            Complete formatted context
        """
        with self._lock:
            context_parts = []

            # Conversation history
            if self.conversation_history:
                context_parts.append("=== CONVERSATION HISTORY ===")
                context_lines = []
                for msg in self.conversation_history:
                    role = msg.role.value.upper()
                    context_lines.append(f"{role}: {msg.content}")
                context_parts.append("\n".join(context_lines))

            # Reasoning trace
            if self.reasoning_trace:
                context_parts.append("\n=== REASONING TRACE ===")
                context_lines = []
                for step in self.reasoning_trace:
                    step_type = step.step_type.value.upper()
                    context_lines.append(f"{step_type}: {step.content}")

                    # Add tool result if present
                    if step.tool_result:
                        result_status = (
                            "SUCCESS" if step.tool_result.success else "FAILED"
                        )
                        context_lines.append(
                            f"  TOOL_RESULT: {step.tool_name} -> {result_status}"
                        )
                context_parts.append("\n".join(context_lines))

            # Tool executions
            if self.tool_executions:
                context_parts.append("\n=== TOOL EXECUTIONS ===")
                context_lines = []
                for exec in self.tool_executions:
                    result_summary = (
                        "SUCCESS"
                        if exec.result.success
                        else f"ERROR: {exec.result.error}"
                    )
                    context_lines.append(
                        f"TOOL: {exec.tool_name}({exec.params}) -> {result_summary}"
                    )
                context_parts.append("\n".join(context_lines))

            return "\n".join(context_parts)

    def clear_session(self) -> None:
        """Clear all memory for new session."""
        with self._lock:
            self.conversation_history.clear()
            self.tool_executions.clear()
            self.reasoning_trace.clear()
            self.last_updated = datetime.now()

    def get_current_problem(self) -> Optional[str]:
        """
        Get the current problem being solved.

        Returns:
            The most recent user message, or None if no user messages exist
        """
        with self._lock:
            for msg in reversed(self.conversation_history):
                if msg.role == MessageRole.USER:
                    return msg.content
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            return {
                "conversation_messages": len(self.conversation_history),
                "tool_executions": len(self.tool_executions),
                "reasoning_steps": len(self.reasoning_trace),
                "session_duration": (datetime.now() - self.created_at).total_seconds(),
                "last_updated": self.last_updated.isoformat(),
                "agent_name": self.agent_name,
                "session_id": self.session_id,
            }

    def _prune_conversation_history(self) -> None:
        """Remove oldest conversation messages to stay within limit."""
        while len(self.conversation_history) > self.max_conversation_length:
            removed = self.conversation_history.pop(0)
            logger.debug(f"Pruned conversation message: {removed.role.value}")

    def _prune_tool_executions(self) -> None:
        """Remove oldest tool executions to stay within limit."""
        while len(self.tool_executions) > self.max_tool_executions:
            removed = self.tool_executions.pop(0)
            logger.debug(f"Pruned tool execution: {removed.tool_name}")

    def _prune_reasoning_trace(self) -> None:
        """Remove oldest reasoning steps to stay within limit."""
        while len(self.reasoning_trace) > self.max_reasoning_steps:
            removed = self.reasoning_trace.pop(0)
            logger.debug(f"Pruned reasoning step: {removed.step_type.value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        with self._lock:
            return {
                "conversation_history": [
                    msg.to_dict() for msg in self.conversation_history
                ],
                "tool_executions": [
                    {
                        "tool_name": exec.tool_name,
                        "params": exec.params,
                        "result": {
                            "success": exec.result.success,
                            "result": exec.result.result,
                            "error": exec.result.error,
                            "execution_time": exec.result.execution_time,
                            "tool_name": exec.result.tool_name,
                        },
                        "timestamp": exec.timestamp.isoformat(),
                    }
                    for exec in self.tool_executions
                ],
                "reasoning_trace": [step.to_dict() for step in self.reasoning_trace],
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "metadata": self.metadata,
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactMemory":
        """Create memory from dictionary."""
        memory = cls(
            agent_name=data.get("agent_name", "ReactAgent"),
            session_id=data.get("session_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
        )

        # Restore conversation history
        for msg_data in data.get("conversation_history", []):
            memory.conversation_history.append(Message.from_dict(msg_data))

        # Restore tool executions
        for exec_data in data.get("tool_executions", []):
            result_data = exec_data["result"]
            result = ToolResult(
                success=result_data["success"],
                result=result_data["result"],
                error=result_data.get("error"),
                execution_time=result_data.get("execution_time", 0.0),
                tool_name=result_data.get("tool_name", ""),
            )
            execution = ToolExecution(
                tool_name=exec_data["tool_name"],
                params=exec_data["params"],
                result=result,
                timestamp=datetime.fromisoformat(exec_data["timestamp"]),
            )
            memory.tool_executions.append(execution)

        # Restore reasoning trace
        for step_data in data.get("reasoning_trace", []):
            memory.reasoning_trace.append(ReasoningStep.from_dict(step_data))

        return memory
