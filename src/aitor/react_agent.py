"""
ReAct Agent implementation building on the Aitor framework.
Combines memory-enabled agents with ReAct reasoning and tool execution.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from .aitor import Aitor
from .llm import BaseLLM, LLMManager
from .memory import ReactMemory, MessageRole
from .reasoning import ReasoningEngine
from .tools import ToolRegistry, Tool, default_registry

logger = logging.getLogger(__name__)


class ReactAgent(Aitor[ReactMemory]):
    """
    ReAct Agent that extends Aitor with reasoning and tool capabilities.

    Combines:
    - Memory-enabled agent architecture from Aitor
    - ReAct reasoning loop (Think -> Act -> Observe)
    - Tool execution and management
    - LLM integration for intelligent responses
    """

    def __init__(
        self,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_reasoning_steps: int = 20,
        max_errors: int = 3,
        memory_config: Optional[Dict[str, Any]] = None,
        task_goal: Optional[str] = None,
        agent_role: Optional[str] = None,
        additional_instructions: Optional[str] = None,
    ):
        """
        Initialize ReAct agent.

        Args:
            name: Agent name
            session_id: Session identifier
            llm: Direct LLM instance
            llm_manager: LLM manager for multiple providers
            tool_registry: Tool registry (uses default if None)
            max_reasoning_steps: Maximum reasoning steps per problem
            max_errors: Maximum errors before stopping
            memory_config: Memory configuration options
            task_goal: Specific task or goal for the agent
            agent_role: Role description for the agent
            additional_instructions: Additional context-specific instructions
        """
        # Initialize memory
        memory_config = memory_config or {}
        initial_memory = ReactMemory(
            agent_name=name or "ReactAgent",
            session_id=session_id or str(uuid.uuid4()),
            **memory_config,
        )

        # Initialize parent Aitor
        super().__init__(
            initial_memory=initial_memory,
            aitor_id=session_id,
            name=name or "ReactAgent",
        )

        # Initialize components
        self.tool_registry = tool_registry or default_registry
        self.llm = llm
        self.llm_manager = llm_manager

        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(
            tool_registry=self.tool_registry,
            llm=llm,
            llm_manager=llm_manager,
            max_reasoning_steps=max_reasoning_steps,
            max_errors=max_errors,
            task_goal=task_goal,
            agent_role=agent_role,
            additional_instructions=additional_instructions,
        )

        logger.info(
            f"Initialized ReactAgent: {self.name} (session: {self.memory.session_id})"
        )

    async def solve(self, problem: str) -> str:
        """
        Solve a problem using ReAct reasoning.

        Args:
            problem: Problem description

        Returns:
            Solution or answer
        """
        logger.info(f"Solving problem: {problem[:100]}...")

        try:
            # Use reasoning engine to solve the problem
            solution = await self.reasoning_engine.solve_problem(problem, self.memory)

            logger.info("Problem solved successfully")
            return solution

        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            error_msg = f"I encountered an error while solving the problem: {str(e)}"
            self.memory.add_message(MessageRole.ASSISTANT, error_msg)
            return error_msg

    async def chat(self, message: str) -> str:
        """
        Chat with the agent (alias for solve).

        Args:
            message: User message

        Returns:
            Agent response
        """
        return await self.solve(message)

    async def on_receive(self, message: str) -> str:
        """
        Handle incoming messages (implements Aitor interface).

        Args:
            message: Incoming message

        Returns:
            Response from agent
        """
        return await self.solve(message)

    async def register_tool(self, tool_instance: Tool) -> None:
        """
        Register a tool with the agent.

        Args:
            tool_instance: Tool to register
        """
        await self.tool_registry.register(tool_instance)
        logger.info(f"Registered tool: {tool_instance.name}")

    async def register_function_as_tool(
        self,
        func,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Register a function as a tool.

        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            **kwargs: Additional tool parameters
        """
        await self.tool_registry.register_function(
            name=name or func.__name__,
            func=func,
            description=description or func.__doc__ or f"Tool: {func.__name__}",
            **kwargs,
        )
        logger.info(f"Registered function as tool: {name or func.__name__}")

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return self.tool_registry.get_tool_names()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all available tools.

        Returns:
            List of tool schemas
        """
        return self.tool_registry.get_tools_schema()

    def get_conversation_history(self) -> str:
        """
        Get formatted conversation history.

        Returns:
            Formatted conversation history
        """
        return self.memory.get_conversation_context()

    def get_reasoning_trace(self) -> str:
        """
        Get formatted reasoning trace.

        Returns:
            Formatted reasoning trace
        """
        return self.memory.get_reasoning_context()

    def get_tool_execution_history(self) -> str:
        """
        Get formatted tool execution history.

        Returns:
            Formatted tool execution history
        """
        return self.memory.get_tool_execution_context()

    def get_full_context(self) -> str:
        """
        Get complete agent context.

        Returns:
            Complete formatted context
        """
        return self.memory.get_full_context()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Memory statistics
        """
        return self.memory.get_stats()

    def clear_session(self) -> None:
        """Clear agent session and memory."""
        self.memory.clear_session()
        logger.info(f"Cleared session for agent: {self.name}")

    def export_memory(self) -> Dict[str, Any]:
        """
        Export agent memory for serialization.

        Returns:
            Serialized memory data
        """
        return self.memory.to_dict()

    def import_memory(self, memory_data: Dict[str, Any]) -> None:
        """
        Import agent memory from serialized data.

        Args:
            memory_data: Serialized memory data
        """
        imported_memory = ReactMemory.from_dict(memory_data)
        self.set_memory(imported_memory)
        logger.info(f"Imported memory for agent: {self.name}")

    def set_llm_provider(self, provider_name: str) -> None:
        """
        Set LLM provider (requires LLM manager).

        Args:
            provider_name: Provider name
        """
        if not self.llm_manager:
            raise ValueError("LLM manager not configured")

        self.llm_manager.set_default(provider_name)
        logger.info(f"Set LLM provider to: {provider_name}")

    def get_current_llm_provider(self) -> str:
        """
        Get current LLM provider name.

        Returns:
            Current provider name
        """
        if self.llm_manager:
            return self.llm_manager.default_provider
        elif self.llm:
            return getattr(self.llm, "model", "direct_llm")
        else:
            return "none"

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Any cleanup needed
        pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ReactAgent(name='{self.name}', "
            f"session='{self.memory.session_id}', "
            f"tools={len(self.tool_registry)}, "
            f"llm='{self.get_current_llm_provider()}')"
        )


class ReactAgentBuilder:
    """
    Builder pattern for creating ReactAgent instances with various configurations.
    """

    def __init__(self):
        """Initialize builder."""
        self._name: Optional[str] = None
        self._session_id: Optional[str] = None
        self._llm: Optional[BaseLLM] = None
        self._llm_manager: Optional[LLMManager] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._max_reasoning_steps: int = 20
        self._max_errors: int = 3
        self._memory_config: Dict[str, Any] = {}
        self._tools_to_register: List[Tool] = []
        self._functions_to_register: List[Dict[str, Any]] = []
        self._task_goal: Optional[str] = None
        self._agent_role: Optional[str] = None
        self._additional_instructions: Optional[str] = None

    def name(self, name: str) -> "ReactAgentBuilder":
        """Set agent name."""
        self._name = name
        return self

    def session_id(self, session_id: str) -> "ReactAgentBuilder":
        """Set session ID."""
        self._session_id = session_id
        return self

    def llm(self, llm: BaseLLM) -> "ReactAgentBuilder":
        """Set LLM instance."""
        self._llm = llm
        return self

    def llm_manager(self, llm_manager: LLMManager) -> "ReactAgentBuilder":
        """Set LLM manager."""
        self._llm_manager = llm_manager
        return self

    def tool_registry(self, tool_registry: ToolRegistry) -> "ReactAgentBuilder":
        """Set tool registry."""
        self._tool_registry = tool_registry
        return self

    def max_reasoning_steps(self, max_steps: int) -> "ReactAgentBuilder":
        """Set maximum reasoning steps."""
        self._max_reasoning_steps = max_steps
        return self

    def max_errors(self, max_errors: int) -> "ReactAgentBuilder":
        """Set maximum errors."""
        self._max_errors = max_errors
        return self

    def task_goal(self, task_goal: str) -> "ReactAgentBuilder":
        """Set task goal."""
        self._task_goal = task_goal
        return self

    def agent_role(self, agent_role: str) -> "ReactAgentBuilder":
        """Set agent role."""
        self._agent_role = agent_role
        return self

    def additional_instructions(self, instructions: str) -> "ReactAgentBuilder":
        """Set additional instructions."""
        self._additional_instructions = instructions
        return self

    def memory_config(self, **config) -> "ReactAgentBuilder":
        """Set memory configuration."""
        self._memory_config.update(config)
        return self

    def add_tool(self, tool_instance: Tool) -> "ReactAgentBuilder":
        """Add a tool to register."""
        self._tools_to_register.append(tool_instance)
        return self

    def add_function_as_tool(
        self,
        func,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> "ReactAgentBuilder":
        """Add a function to register as a tool."""
        self._functions_to_register.append(
            {"func": func, "name": name, "description": description, **kwargs}
        )
        return self

    async def build(self) -> ReactAgent:
        """
        Build the ReactAgent instance.

        Returns:
            Configured ReactAgent
        """
        # Create agent
        agent = ReactAgent(
            name=self._name,
            session_id=self._session_id,
            llm=self._llm,
            llm_manager=self._llm_manager,
            tool_registry=self._tool_registry,
            max_reasoning_steps=self._max_reasoning_steps,
            max_errors=self._max_errors,
            memory_config=self._memory_config,
            task_goal=self._task_goal,
            agent_role=self._agent_role,
            additional_instructions=self._additional_instructions,
        )

        # Register tools
        for tool_instance in self._tools_to_register:
            await agent.register_tool(tool_instance)

        # Register functions as tools
        for func_config in self._functions_to_register:
            await agent.register_function_as_tool(**func_config)

        return agent


# Convenience function for creating agents
async def create_react_agent(
    name: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Union[Tool, Callable]]] = None,
    **kwargs,
) -> ReactAgent:
    """
    Convenience function to create a ReactAgent.

    Args:
        name: Agent name
        llm_provider: LLM provider name
        llm_config: LLM configuration
        tools: List of tools or functions to register
        **kwargs: Additional agent configuration

    Returns:
        Configured ReactAgent
    """
    builder = ReactAgentBuilder()

    if name:
        builder.name(name)

    # Set up LLM if provided
    if llm_provider and llm_config:
        from .llm import LLMFactory

        llm = LLMFactory.create(llm_provider, llm_config)
        builder.llm(llm)

    # Add tools
    if tools:
        for tool_item in tools:
            if isinstance(tool_item, Tool):
                builder.add_tool(tool_item)
            elif callable(tool_item):
                builder.add_function_as_tool(tool_item)

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(builder, key):
            getattr(builder, key)(value)

    return await builder.build()
