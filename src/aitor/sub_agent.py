"""
Sub-agent system for bounded task execution.
Provides isolated agents that can execute specific tasks without affecting main context.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

from .llm import BaseLLM, LLMManager, Message as LLMMessage, AgentAction, ThinkAction, ToolAction, ResultAction
from .memory import ReactMemory, Message, MessageRole, ReasoningStep, ReasoningStepType
from .tools import ToolRegistry, ToolResult, ToolExecution
from .todo import TodoItem, TodoStatus

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Result of sub-agent execution."""
    
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubAgent:
    """
    Isolated agent for executing bounded tasks.
    
    Features:
    - Isolated memory and context
    - Limited reasoning steps
    - Tool access (inherited from parent)
    - Focused system prompt
    - Result summarization
    """
    
    def __init__(
        self,
        name: str,
        tool_registry: ToolRegistry,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None,
        max_reasoning_steps: int = 10,
        max_errors: int = 2,
        timeout: float = 60.0,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize sub-agent.
        
        Args:
            name: Name of the sub-agent
            tool_registry: Registry of available tools
            llm: Direct LLM instance to use
            llm_manager: LLM manager for multiple providers
            max_reasoning_steps: Maximum reasoning steps
            max_errors: Maximum errors before stopping
            timeout: Maximum execution time in seconds
            system_prompt: Custom system prompt
        """
        self.name = name
        self.id = str(uuid4())
        self.tool_registry = tool_registry
        self.llm = llm
        self.llm_manager = llm_manager
        self.max_reasoning_steps = max_reasoning_steps
        self.max_errors = max_errors
        self.timeout = timeout
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Isolated memory for this sub-agent
        self.memory = ReactMemory()
        
        if not self.llm and not self.llm_manager:
            raise ValueError("Either llm or llm_manager must be provided")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for sub-agent."""
        return f"""You are {self.name}, a focused sub-agent designed to execute specific tasks efficiently.

CRITICAL: You MUST respond with valid JSON only. No other text allowed.

Response format - choose ONE action type per response:

1. THINKING (brief reasoning):
{{
    "action": "think",
    "content": "brief reasoning about the task"
}}

2. TOOL USAGE (when you need to use a tool):
{{
    "action": "use_tool",
    "tool_name": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}

3. FINAL RESULT (when task is complete):
{{
    "action": "result",
    "content": "final answer or result"
}}

Examples:
- Task: "Calculate 5+3"
  Response: {{"action": "use_tool", "tool_name": "calculator", "parameters": {{"expression": "5+3"}}}}

- Task: "Get current time"  
  Response: {{"action": "use_tool", "tool_name": "current_time", "parameters": {{}}}}

- Task: "Simple question"
  Response: {{"action": "think", "content": "analyzing the question"}}

RULES:
1. ALWAYS respond with valid JSON
2. Be direct and efficient - you have limited steps  
3. Use tools for calculations or data retrieval
4. Provide result when you have the final answer
5. No explanatory text outside JSON"""
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> SubAgentResult:
        """
        Execute a specific task with isolated context.
        
        Args:
            task: Task description to execute
            context: Optional context information
            
        Returns:
            SubAgentResult with execution details
        """
        start_time = datetime.now()
        
        logger.info(f"SubAgent {self.name} executing task: {task}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task_internal(task, context),
                timeout=self.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"SubAgent {self.name} completed task in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"SubAgent {self.name} timed out after {execution_time:.2f}s")
            
            return SubAgentResult(
                success=False,
                error=f"Task execution timed out after {self.timeout}s",
                execution_time=execution_time,
                reasoning_steps=self.memory.reasoning_trace,
                tool_executions=self.memory.tool_executions
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"SubAgent {self.name} failed: {str(e)}", exc_info=True)
            
            return SubAgentResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                reasoning_steps=self.memory.reasoning_trace,
                tool_executions=self.memory.tool_executions
            )
    
    async def _execute_task_internal(self, task: str, context: Optional[Dict[str, Any]]) -> SubAgentResult:
        """Internal task execution logic."""
        # Clear previous memory
        self.memory = ReactMemory()
        
        # Add task and context to memory
        self.memory.add_message(MessageRole.USER, task)
        
        if context:
            context_msg = f"Context: {context}"
            self.memory.add_message(MessageRole.SYSTEM, context_msg)
        
        error_count = 0
        step_count = 0
        
        while step_count < self.max_reasoning_steps and error_count < self.max_errors:
            try:
                # Generate reasoning step
                reasoning_step = await self._generate_reasoning_step()
                self.memory.add_reasoning_step(reasoning_step)
                
                logger.info(f"SubAgent {self.name} step {step_count + 1}: {reasoning_step.step_type.value} - {reasoning_step.content[:100]}...")
                
                # Handle different step types
                if reasoning_step.step_type == ReasoningStepType.THINK:
                    # Continue thinking
                    logger.debug(f"SubAgent {self.name} thinking: {reasoning_step.content}")
                    
                elif reasoning_step.step_type == ReasoningStepType.ACT:
                    # Execute tool
                    if reasoning_step.tool_name and reasoning_step.tool_params is not None:
                        logger.debug(f"SubAgent {self.name} executing tool: {reasoning_step.tool_name}")
                        tool_result = await self._execute_tool(
                            reasoning_step.tool_name,
                            reasoning_step.tool_params
                        )
                        reasoning_step.tool_result = tool_result
                        
                        # Create observation
                        observation = self._create_observation(
                            reasoning_step.tool_name,
                            tool_result
                        )
                        self.memory.add_reasoning_step(observation)
                        
                        # For simple calculation tasks, if we get a successful result,
                        # we can often complete the task immediately
                        if tool_result.success and "calculator" in reasoning_step.tool_name.lower():
                            return SubAgentResult(
                                success=True,
                                result=str(tool_result.result),
                                reasoning_steps=self.memory.reasoning_trace,
                                tool_executions=self.memory.tool_executions
                            )
                    
                elif reasoning_step.step_type == ReasoningStepType.FINAL_ANSWER:
                    # Task completed
                    return SubAgentResult(
                        success=True,
                        result=reasoning_step.content,
                        reasoning_steps=self.memory.reasoning_trace,
                        tool_executions=self.memory.tool_executions
                    )
                
                # Check for RESULT format (sub-agent specific) - this should be handled by parsing
                # but adding as backup check
                if "RESULT:" in reasoning_step.content.upper():
                    result_index = reasoning_step.content.upper().find("RESULT:")
                    result_content = reasoning_step.content[result_index + 7:].strip()
                    
                    return SubAgentResult(
                        success=True,
                        result=result_content,
                        reasoning_steps=self.memory.reasoning_trace,
                        tool_executions=self.memory.tool_executions
                    )
                
                step_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"SubAgent {self.name} error in step {step_count + 1}: {str(e)}")
                
                # Add error to reasoning trace
                error_step = ReasoningStep(
                    step_type=ReasoningStepType.OBSERVE,
                    content=f"Error occurred: {str(e)}. Adjusting approach.",
                    metadata={"error": str(e), "step_count": step_count}
                )
                self.memory.add_reasoning_step(error_step)
                
                if error_count >= self.max_errors:
                    break
        
        # If we reach here, we've exhausted steps or errors
        failure_reason = "Task execution incomplete"
        if step_count >= self.max_reasoning_steps:
            failure_reason += f" (reached max steps: {self.max_reasoning_steps})"
        if error_count >= self.max_errors:
            failure_reason += f" (too many errors: {error_count})"
        
        return SubAgentResult(
            success=False,
            error=failure_reason,
            reasoning_steps=self.memory.reasoning_trace,
            tool_executions=self.memory.tool_executions
        )
    
    async def _generate_reasoning_step(self) -> ReasoningStep:
        """Generate next reasoning step using LLM with structured output."""
        # Build messages
        messages = self._build_llm_messages()
        
        # Get structured LLM response using Pydantic models
        if self.llm:
            action_response = await self.llm.generate_structured(messages, AgentAction)
        else:
            action_response = await self.llm_manager.generate_structured(messages, AgentAction)
        
        # Debug logging
        logger.info(f"SubAgent {self.name} structured response: {action_response}")
        
        # Convert structured response to ReasoningStep
        return self._convert_action_to_reasoning_step(action_response)
    
    def _build_llm_messages(self) -> List[LLMMessage]:
        """Build LLM messages from memory."""
        messages = []
        
        # System message with tools info
        system_content = self.system_prompt
        tools_info = self._format_tools_info()
        if tools_info:
            system_content += "\n\n" + tools_info
        
        messages.append(LLMMessage("system", system_content))
        
        # Add conversation history
        for msg in self.memory.conversation_history:
            messages.append(LLMMessage(msg.role.value, msg.content))
        
        # Add recent reasoning context
        if self.memory.reasoning_trace:
            context_parts = []
            for step in self.memory.reasoning_trace[-5:]:  # Last 5 steps
                if step.step_type == ReasoningStepType.THINK:
                    context_parts.append(f"THINK: {step.content}")
                elif step.step_type == ReasoningStepType.ACT:
                    context_parts.append(f"ACT: {step.content}")
                elif step.step_type == ReasoningStepType.OBSERVE:
                    context_parts.append(f"OBSERVE: {step.content}")
            
            if context_parts:
                context_message = "Recent reasoning:\n" + "\n".join(context_parts)
                messages.append(LLMMessage("assistant", context_message))
        
        # Prompt for next step
        messages.append(LLMMessage("user", "Continue with your reasoning. What is your next step?"))
        
        return messages
    
    def _format_tools_info(self) -> str:
        """Format available tools information."""
        tools = self.tool_registry.get_all_tools()
        if not tools:
            return ""
        
        tool_descriptions = ["AVAILABLE TOOLS:"]
        for tool in tools:
            param_parts = []
            for param_name, param_info in tool.parameters.items():
                param_type = param_info.get('type', 'Any')
                required = param_info.get('required', False)
                param_str = f"{param_name}: {param_type}"
                if not required:
                    param_str += " (optional)"
                param_parts.append(param_str)
            
            params_str = ", ".join(param_parts)
            tool_descriptions.append(f"- {tool.name}({params_str}): {tool.description}")
        
        return "\n".join(tool_descriptions)
    
    async def _execute_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> ToolResult:
        """Execute a tool and record in memory."""
        logger.debug(f"SubAgent {self.name} executing tool: {tool_name}")
        
        tool_result = await self.tool_registry.execute_tool(tool_name, **tool_params)
        
        # Record tool execution
        tool_execution = ToolExecution(
            tool_name=tool_name,
            params=tool_params,
            result=tool_result,
            timestamp=datetime.now()
        )
        self.memory.add_tool_execution(tool_execution)
        
        return tool_result
    
    def _create_observation(self, tool_name: str, tool_result: ToolResult) -> ReasoningStep:
        """Create observation step from tool result."""
        if tool_result.success:
            content = f"Tool '{tool_name}' executed successfully. Result: {tool_result.result}"
        else:
            content = f"Tool '{tool_name}' failed with error: {tool_result.error}"
        
        return ReasoningStep(
            step_type=ReasoningStepType.OBSERVE,
            content=content,
            tool_name=tool_name,
            tool_result=tool_result
        )
    
    def _convert_action_to_reasoning_step(self, action: AgentAction) -> ReasoningStep:
        """Convert structured action to reasoning step."""
        try:
            if isinstance(action, ThinkAction):
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=action.content
                )
            
            elif isinstance(action, ToolAction):
                if not action.tool_name:
                    raise ValueError("Tool name not specified in use_tool action")
                
                # Create action description for content
                param_strs = [f'{k}="{v}"' for k, v in action.parameters.items()]
                action_content = f"{action.tool_name}({', '.join(param_strs)})"
                
                return ReasoningStep(
                    step_type=ReasoningStepType.ACT,
                    content=action_content,
                    tool_name=action.tool_name,
                    tool_params=action.parameters
                )
            
            elif isinstance(action, ResultAction):
                return ReasoningStep(
                    step_type=ReasoningStepType.FINAL_ANSWER,
                    content=action.content
                )
            
            else:
                # Unknown action type, treat as thinking
                logger.warning(f"Unknown action type '{type(action)}', treating as thinking")
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Failed to convert action to reasoning step: {e}, action: {action}")
            # Fallback to thinking step
            return ReasoningStep(
                step_type=ReasoningStepType.THINK,
                content=f"Failed to process action: {str(action)[:100]}"
            )


class SubAgentManager:
    """
    Manages sub-agents for different types of tasks.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None
    ):
        self.tool_registry = tool_registry
        self.llm = llm
        self.llm_manager = llm_manager
        self.sub_agents: Dict[str, SubAgent] = {}
    
    def create_sub_agent(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        max_reasoning_steps: int = 10,
        max_errors: int = 2,
        timeout: float = 60.0
    ) -> SubAgent:
        """Create a new sub-agent."""
        sub_agent = SubAgent(
            name=name,
            tool_registry=self.tool_registry,
            llm=self.llm,
            llm_manager=self.llm_manager,
            max_reasoning_steps=max_reasoning_steps,
            max_errors=max_errors,
            timeout=timeout,
            system_prompt=system_prompt
        )
        
        self.sub_agents[name] = sub_agent
        logger.info(f"Created sub-agent: {name}")
        return sub_agent
    
    def get_sub_agent(self, name: str) -> Optional[SubAgent]:
        """Get sub-agent by name."""
        return self.sub_agents.get(name)
    
    async def execute_with_sub_agent(
        self,
        task: str,
        agent_name: str = "default",
        context: Optional[Dict[str, Any]] = None,
        **sub_agent_kwargs
    ) -> SubAgentResult:
        """Execute task with a sub-agent."""
        sub_agent = self.sub_agents.get(agent_name)
        
        if not sub_agent:
            # Create default sub-agent
            sub_agent = self.create_sub_agent(agent_name, **sub_agent_kwargs)
        
        return await sub_agent.execute_task(task, context)
    
    def list_sub_agents(self) -> List[str]:
        """List all sub-agent names."""
        return list(self.sub_agents.keys())
    
    def clear_sub_agents(self):
        """Clear all sub-agents."""
        self.sub_agents.clear()
        logger.info("Cleared all sub-agents")