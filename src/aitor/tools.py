"""
Tool system for ReAct agents.
Provides tool registration, execution, and management capabilities.
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    
    def __post_init__(self):
        if not self.success and self.error is None:
            self.error = "Unknown error occurred"


@dataclass
class ToolExecution:
    """Metadata about a tool execution."""
    tool_name: str
    params: Dict[str, Any]
    result: ToolResult
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Tool:
    """Base class for all tools available to ReAct agents."""
    
    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        async_execution: bool = True
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name for identification
            func: Function to execute
            description: Human-readable description
            parameters: JSON schema for parameters (optional)
            timeout: Execution timeout in seconds
            async_execution: Whether to execute in thread pool
        """
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters or self._extract_parameters()
        self.timeout = timeout
        self.async_execution = async_execution
        self.signature = inspect.signature(func)
        
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter information from function signature."""
        sig = inspect.signature(self.func)
        params = {}
        
        for param_name, param in sig.parameters.items():
            param_info = {
                "name": param_name,
                "required": param.default == inspect.Parameter.empty,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            }
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            params[param_name] = param_info
            
        return params
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Parameters to pass to the tool function
            
        Returns:
            ToolResult containing execution outcome
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            self._validate_parameters(kwargs)
            
            # Execute function
            if self.async_execution:
                if asyncio.iscoroutinefunction(self.func):
                    result = await asyncio.wait_for(
                        self.func(**kwargs),
                        timeout=self.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.func, **kwargs),
                        timeout=self.timeout
                    )
            else:
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(**kwargs)
                else:
                    result = self.func(**kwargs)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Tool {self.name} executed successfully in {execution_time:.2f}s")
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                tool_name=self.name
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Tool {self.name} timed out after {self.timeout}s"
            logger.error(error_msg)
            
            return ToolResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                tool_name=self.name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool {self.name} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ToolResult(
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                tool_name=self.name
            )
    
    def _validate_parameters(self, kwargs: Dict[str, Any]) -> None:
        """Validate parameters against function signature."""
        sig = self.signature
        
        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        # Check for unexpected parameters
        for param_name in kwargs:
            if param_name not in sig.parameters:
                raise ValueError(f"Unexpected parameter: {param_name}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout": self.timeout
        }
    
    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')"


class ToolRegistry:
    """Registry for managing tools available to ReAct agents."""
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, Tool] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool to register
            
        Raises:
            ValueError: If tool name already exists
        """
        async with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            
            self._tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")
    
    async def register_function(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        async_execution: bool = True
    ) -> None:
        """
        Register a function as a tool.
        
        Args:
            name: Tool name
            func: Function to register
            description: Tool description
            parameters: Parameter schema
            timeout: Execution timeout
            async_execution: Whether to execute async
        """
        tool = Tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            timeout=timeout,
            async_execution=async_execution
        )
        await self.register(tool)
    
    async def unregister(self, name: str) -> None:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
        """
        async with self._lock:
            if name not in self._tools:
                logger.warning(f"Attempted to unregister non-existent tool: {name}")
                return
            
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get list of all registered tools."""
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get schema for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Parameters for the tool
            
        Returns:
            ToolResult containing execution outcome
        """
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{name}' not found",
                tool_name=name
            )
        
        return await tool.execute(**kwargs)
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    async_execution: bool = True
):
    """
    Decorator to create a tool from a function.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        parameters: Parameter schema
        timeout: Execution timeout
        async_execution: Whether to execute async
        
    Returns:
        Tool object
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"
        
        return Tool(
            name=tool_name,
            func=func,
            description=tool_description,
            parameters=parameters,
            timeout=timeout,
            async_execution=async_execution
        )
    
    return decorator


# Global tool registry instance
default_registry = ToolRegistry()