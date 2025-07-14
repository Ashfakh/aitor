"""
Planning-based ReAct Agent with todo management and sub-agents.
Enhanced version of ReactAgent with Claude Code-like planning capabilities.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from .aitor import Aitor
from .llm import BaseLLM, LLMManager
from .memory import ReactMemory, MessageRole
from .planning_reasoning import PlanningReasoningEngine
from .tools import ToolRegistry, Tool, tool, default_registry
from .todo import TodoManager, TodoItem, TodoStatus
from .sub_agent import SubAgentManager

logger = logging.getLogger(__name__)


class PlanningReactAgent(Aitor[ReactMemory]):
    """
    Planning-enabled ReAct Agent with todo management and sub-agents.
    
    Features:
    - Task planning and breakdown
    - Todo management and tracking
    - Sub-agent delegation for bounded tasks
    - Progress monitoring and adaptation
    - Enhanced reasoning with planning
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_reasoning_steps: int = 50,
        max_errors: int = 5,
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Planning ReAct agent.
        
        Args:
            name: Agent name
            session_id: Session identifier
            llm: Direct LLM instance
            llm_manager: LLM manager for multiple providers
            tool_registry: Tool registry (uses default if None)
            system_prompt: Custom system prompt
            max_reasoning_steps: Maximum reasoning steps per problem
            max_errors: Maximum errors before stopping
            memory_config: Memory configuration options
        """
        # Initialize memory
        memory_config = memory_config or {}
        initial_memory = ReactMemory(
            agent_name=name or "PlanningReactAgent",
            session_id=session_id or str(uuid.uuid4()),
            **memory_config
        )
        
        # Initialize parent Aitor
        super().__init__(
            initial_memory=initial_memory,
            aitor_id=session_id,
            name=name or "PlanningReactAgent"
        )
        
        # Initialize components
        self.tool_registry = tool_registry or ToolRegistry()
        self.llm = llm
        self.llm_manager = llm_manager
        
        # Initialize planning reasoning engine
        self.reasoning_engine = PlanningReasoningEngine(
            tool_registry=self.tool_registry,
            llm=llm,
            llm_manager=llm_manager,
            max_reasoning_steps=max_reasoning_steps,
            max_errors=max_errors,
            system_prompt=system_prompt
        )
        
        logger.info(f"Initialized PlanningReactAgent: {self.name} (session: {self.memory.session_id})")
    
    async def solve(self, problem: str) -> str:
        """
        Solve a problem using planning-based reasoning.
        
        Args:
            problem: Problem description
            
        Returns:
            Solution or answer
        """
        logger.info(f"Solving problem with planning: {problem[:100]}...")
        
        try:
            # Clear previous planning state
            await self.clear_planning_state()
            
            # Use planning reasoning engine to solve the problem
            solution = await self.reasoning_engine.solve_problem(problem, self.memory)
            
            logger.info(f"Problem solved successfully with planning")
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
        **kwargs
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
            **kwargs
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
    
    # Planning-specific methods
    
    def get_todo_manager(self) -> TodoManager:
        """
        Get the todo manager instance.
        
        Returns:
            TodoManager instance
        """
        return self.reasoning_engine.get_todo_manager()
    
    def get_sub_agent_manager(self) -> SubAgentManager:
        """
        Get the sub-agent manager instance.
        
        Returns:
            SubAgentManager instance
        """
        return self.reasoning_engine.get_sub_agent_manager()
    
    async def get_current_todos(self) -> List[TodoItem]:
        """
        Get all current todos.
        
        Returns:
            List of TodoItems
        """
        return self.get_todo_manager().get_all_todos()
    
    async def get_todos_by_status(self, status: TodoStatus) -> List[TodoItem]:
        """
        Get todos by status.
        
        Args:
            status: TodoStatus to filter by
            
        Returns:
            List of TodoItems with specified status
        """
        return await self.get_todo_manager().get_todos_by_status(status)
    
    async def get_planning_progress(self) -> Dict[str, Any]:
        """
        Get current planning progress.
        
        Returns:
            Planning progress summary
        """
        return await self.reasoning_engine.get_planning_summary()
    
    async def get_todo_tree(self) -> Dict[str, Any]:
        """
        Get todos organized in tree structure.
        
        Returns:
            Todo tree structure
        """
        return self.get_todo_manager().get_todo_tree()
    
    async def clear_planning_state(self):
        """Clear all planning state (todos, sub-agents)."""
        await self.get_todo_manager().clear_todos()
        self.get_sub_agent_manager().clear_sub_agents()
        logger.info("Cleared planning state")
    
    def get_sub_agents(self) -> List[str]:
        """
        Get list of active sub-agents.
        
        Returns:
            List of sub-agent names
        """
        return self.get_sub_agent_manager().list_sub_agents()
    
    # Enhanced context methods
    
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
        Get complete agent context including planning state.
        
        Returns:
            Complete formatted context
        """
        base_context = self.memory.get_full_context()
        
        # Add planning context
        todos = self.get_todo_manager().get_all_todos()
        if todos:
            planning_context = "\n=== PLANNING CONTEXT ===\n"
            planning_context += f"Total todos: {len(todos)}\n"
            
            # Group by status
            by_status = {}
            for todo in todos:
                status = todo.status.value
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(todo)
            
            for status, todo_list in by_status.items():
                planning_context += f"\n{status.upper()} ({len(todo_list)}):\n"
                for todo in todo_list:
                    planning_context += f"  - {todo.content}\n"
                    if todo.result:
                        planning_context += f"    Result: {todo.result}\n"
                    if todo.error:
                        planning_context += f"    Error: {todo.error}\n"
            
            # Add sub-agents
            sub_agents = self.get_sub_agents()
            if sub_agents:
                planning_context += f"\nActive sub-agents: {', '.join(sub_agents)}\n"
            
            base_context += planning_context
        
        return base_context
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics including planning stats.
        
        Returns:
            Extended memory statistics
        """
        stats = self.memory.get_stats()
        
        # Add planning stats
        todos = self.get_todo_manager().get_all_todos()
        stats["planning"] = {
            "total_todos": len(todos),
            "completed_todos": len([t for t in todos if t.status == TodoStatus.COMPLETED]),
            "pending_todos": len([t for t in todos if t.status == TodoStatus.PENDING]),
            "failed_todos": len([t for t in todos if t.status == TodoStatus.FAILED]),
            "sub_agents": len(self.get_sub_agents())
        }
        
        return stats
    
    def clear_session(self) -> None:
        """Clear agent session and memory."""
        self.memory.clear_session()
        asyncio.create_task(self.clear_planning_state())
        logger.info(f"Cleared session for planning agent: {self.name}")
    
    def export_memory(self) -> Dict[str, Any]:
        """
        Export agent memory including planning state.
        
        Returns:
            Serialized memory and planning data
        """
        memory_data = self.memory.to_dict()
        
        # Add planning state
        memory_data["planning"] = {
            "todos": [todo.to_dict() for todo in self.get_todo_manager().get_all_todos()],
            "sub_agents": self.get_sub_agents()
        }
        
        return memory_data
    
    def import_memory(self, memory_data: Dict[str, Any]) -> None:
        """
        Import agent memory including planning state.
        
        Args:
            memory_data: Serialized memory and planning data
        """
        # Import base memory
        planning_data = memory_data.pop("planning", {})
        imported_memory = ReactMemory.from_dict(memory_data)
        self.set_memory(imported_memory)
        
        # Import planning state
        if planning_data:
            # Clear current state
            asyncio.create_task(self.clear_planning_state())
            
            # Import todos
            for todo_data in planning_data.get("todos", []):
                todo = TodoItem.from_dict(todo_data)
                self.get_todo_manager().todos[todo.id] = todo
                self.get_todo_manager().execution_order.append(todo.id)
        
        logger.info(f"Imported memory and planning state for agent: {self.name}")
    
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
            return getattr(self.llm, 'model', 'direct_llm')
        else:
            return "none"
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.clear_planning_state()
    
    def __repr__(self) -> str:
        """String representation."""
        todos = self.get_todo_manager().get_all_todos()
        return (
            f"PlanningReactAgent(name='{self.name}', "
            f"session='{self.memory.session_id}', "
            f"tools={len(self.tool_registry)}, "
            f"todos={len(todos)}, "
            f"sub_agents={len(self.get_sub_agents())}, "
            f"llm='{self.get_current_llm_provider()}')"
        )


class PlanningReactAgentBuilder:
    """
    Builder pattern for creating PlanningReactAgent instances.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._name: Optional[str] = None
        self._session_id: Optional[str] = None
        self._llm: Optional[BaseLLM] = None
        self._llm_manager: Optional[LLMManager] = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._system_prompt: Optional[str] = None
        self._max_reasoning_steps: int = 50
        self._max_errors: int = 5
        self._memory_config: Dict[str, Any] = {}
        self._tools_to_register: List[Tool] = []
        self._functions_to_register: List[Dict[str, Any]] = []
    
    def name(self, name: str) -> 'PlanningReactAgentBuilder':
        """Set agent name."""
        self._name = name
        return self
    
    def session_id(self, session_id: str) -> 'PlanningReactAgentBuilder':
        """Set session ID."""
        self._session_id = session_id
        return self
    
    def llm(self, llm: BaseLLM) -> 'PlanningReactAgentBuilder':
        """Set LLM instance."""
        self._llm = llm
        return self
    
    def llm_manager(self, llm_manager: LLMManager) -> 'PlanningReactAgentBuilder':
        """Set LLM manager."""
        self._llm_manager = llm_manager
        return self
    
    def tool_registry(self, tool_registry: ToolRegistry) -> 'PlanningReactAgentBuilder':
        """Set tool registry."""
        self._tool_registry = tool_registry
        return self
    
    def system_prompt(self, system_prompt: str) -> 'PlanningReactAgentBuilder':
        """Set system prompt."""
        self._system_prompt = system_prompt
        return self
    
    def max_reasoning_steps(self, max_steps: int) -> 'PlanningReactAgentBuilder':
        """Set maximum reasoning steps."""
        self._max_reasoning_steps = max_steps
        return self
    
    def max_errors(self, max_errors: int) -> 'PlanningReactAgentBuilder':
        """Set maximum errors."""
        self._max_errors = max_errors
        return self
    
    def memory_config(self, **config) -> 'PlanningReactAgentBuilder':
        """Set memory configuration."""
        self._memory_config.update(config)
        return self
    
    def add_tool(self, tool_instance: Tool) -> 'PlanningReactAgentBuilder':
        """Add a tool to register."""
        self._tools_to_register.append(tool_instance)
        return self
    
    def add_function_as_tool(self, func, name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> 'PlanningReactAgentBuilder':
        """Add a function to register as a tool."""
        self._functions_to_register.append({
            'func': func,
            'name': name,
            'description': description,
            **kwargs
        })
        return self
    
    async def build(self) -> PlanningReactAgent:
        """
        Build the PlanningReactAgent instance.
        
        Returns:
            Configured PlanningReactAgent
        """
        # Create agent
        agent = PlanningReactAgent(
            name=self._name,
            session_id=self._session_id,
            llm=self._llm,
            llm_manager=self._llm_manager,
            tool_registry=self._tool_registry,
            system_prompt=self._system_prompt,
            max_reasoning_steps=self._max_reasoning_steps,
            max_errors=self._max_errors,
            memory_config=self._memory_config
        )
        
        # Register tools
        for tool_instance in self._tools_to_register:
            await agent.register_tool(tool_instance)
        
        # Register functions as tools
        for func_config in self._functions_to_register:
            await agent.register_function_as_tool(**func_config)
        
        return agent


# Convenience function for creating planning agents
async def create_planning_agent(
    name: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Union[Tool, callable]]] = None,
    **kwargs
) -> PlanningReactAgent:
    """
    Convenience function to create a PlanningReactAgent.
    
    Args:
        name: Agent name
        llm_provider: LLM provider name
        llm_config: LLM configuration
        tools: List of tools or functions to register
        **kwargs: Additional agent configuration
        
    Returns:
        Configured PlanningReactAgent
    """
    builder = PlanningReactAgentBuilder()
    
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