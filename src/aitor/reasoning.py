"""
ReAct reasoning engine for intelligent agents.
Implements the Think -> Act -> Observe reasoning loop.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .llm import BaseLLM, LLMManager, Message as LLMMessage
from pydantic import BaseModel, Field
from typing import Literal
from .memory import ReactMemory, ReasoningStep, ReasoningStepType, MessageRole
from .tools import ToolRegistry, ToolResult, ToolExecution
from .logging_config import log_prompt, log_response, log_reasoning_step, log_tool_execution, log_section_start, log_section_end

logger = logging.getLogger(__name__)


# Comprehensive response model for React agents
class ReactAgentResponse(BaseModel):
    """Comprehensive response model that handles all action types."""
    action: Literal["think", "use_tool", "result"] = Field(..., description="Type of action to take")
    content: Optional[str] = Field(None, description="Content for think or result actions")
    tool_name: Optional[str] = Field(None, description="Name of tool to use (for use_tool action)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for tool (for use_tool action)")




class ReasoningEngine:
    """
    Core ReAct reasoning engine.
    
    Orchestrates the Think -> Act -> Observe loop for intelligent problem solving.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None,
        max_reasoning_steps: int = 20,
        max_errors: int = 3,
        system_prompt: Optional[str] = None,
        task_goal: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        agent_role: Optional[str] = None
    ):
        """
        Initialize reasoning engine.
        
        Args:
            tool_registry: Registry of available tools
            llm: Direct LLM instance to use
            llm_manager: LLM manager for multiple providers
            max_reasoning_steps: Maximum reasoning steps per problem
            max_errors: Maximum errors before stopping
            system_prompt: Custom system prompt for the agent (if provided, overrides default)
            task_goal: Specific task or goal for the agent
            additional_instructions: Additional context-specific instructions
            agent_role: Role description for the agent (e.g., "calculator_agent")
        """
        self.tool_registry = tool_registry
        self.llm = llm
        self.llm_manager = llm_manager
        self.max_reasoning_steps = max_reasoning_steps
        self.max_errors = max_errors
        self.task_goal = task_goal
        self.additional_instructions = additional_instructions
        self.agent_role = agent_role
        self.custom_system_prompt = system_prompt
        
        if not self.llm and not self.llm_manager:
            raise ValueError("Either llm or llm_manager must be provided")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from components."""
        # Base ReAct prompt with structured response format
        base_prompt = self._get_base_react_prompt()
        
        # Add task-specific information
        if self.task_goal:
            base_prompt += f"\n\nYour current task: {self.task_goal}"
        
        if self.agent_role:
            base_prompt = base_prompt.replace(
                "You are a ReAct (Reasoning and Acting) agent",
                f"You are a {self.agent_role} using ReAct (Reasoning and Acting) methodology"
            )
        
        if self.additional_instructions:
            base_prompt += f"\n\nAdditional instructions:\n{self.additional_instructions}"
        
        # Add available tools information
        tools_info = self._format_tools_info()
        if tools_info:
            base_prompt += "\n\n" + tools_info
        
        return base_prompt
    
    def _get_base_react_prompt(self) -> str:
        """Get base ReAct prompt with structured response format."""
        return """You are a ReAct (Reasoning and Acting) agent. Your task is to solve problems by thinking step by step and using available tools when needed.

You operate in a loop of Thought, Action, and Observation:
1. Thought: Analyze the problem and plan your approach
2. Action: Use tools to gather information or perform tasks
3. Observation: Analyze the results and continue reasoning

You must respond with a structured JSON object with the following schema:

Required field:
- "action": Must be one of "think", "use_tool", or "result"

Optional fields (populate based on action type):
- "content": For "think" and "result" actions - your reasoning or final answer
- "tool_name": For "use_tool" action - exact name of the tool to use
- "parameters": For "use_tool" action - object with tool parameters

Examples:

For thinking/reasoning:
{"action": "think", "content": "Your reasoning about the problem and what to do next"}

For using a tool:
{"action": "use_tool", "tool_name": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}

For providing the final answer:
{"action": "result", "content": "Your complete answer to the user's question"}

Important:
- Be thorough in your thinking before taking actions
- Always use tools when you need specific information or to perform tasks
- Analyze observations carefully before proceeding
- Provide a clear, complete final answer when you have enough information
- If you encounter errors, adapt your approach and try alternatives
- Always respond with valid JSON matching the schema above"""
    
    async def solve_problem(self, problem: str, memory: ReactMemory) -> str:
        """
        Solve a problem using ReAct reasoning.
        
        Args:
            problem: Problem description
            memory: Agent memory
            
        Returns:
            Final answer or solution
        """
        log_section_start(f"ReAct Reasoning: {problem[:50]}...")
        
        # Add problem to memory
        memory.add_message(MessageRole.USER, problem)
        
        error_count = 0
        step_count = 0
        
        while step_count < self.max_reasoning_steps and error_count < self.max_errors:
            try:
                # Generate reasoning step
                reasoning_step = await self._generate_reasoning_step(memory)
                memory.add_reasoning_step(reasoning_step)
                
                # Log reasoning step
                log_reasoning_step(
                    "ReAct Engine",
                    reasoning_step.step_type.value,
                    reasoning_step.content,
                    {"step_count": step_count + 1, "tool_name": reasoning_step.tool_name}
                )
                
                # Handle different step types
                if reasoning_step.step_type == ReasoningStepType.THINK:
                    # Thinking step - continue to next iteration
                    pass
                    
                elif reasoning_step.step_type == ReasoningStepType.ACT:
                    # Action step - execute tool
                    if reasoning_step.tool_name and reasoning_step.tool_params is not None:
                        tool_result = await self._execute_tool_action(
                            reasoning_step.tool_name,
                            reasoning_step.tool_params,
                            memory
                        )
                        reasoning_step.tool_result = tool_result
                        
                        # Create observation step
                        observation_step = await self._create_observation_step(
                            reasoning_step.tool_name,
                            tool_result,
                            memory
                        )
                        memory.add_reasoning_step(observation_step)
                        
                        # Log observation
                        log_reasoning_step(
                            "ReAct Engine",
                            observation_step.step_type.value,
                            observation_step.content,
                            {"tool_result": tool_result.success}
                        )
                    else:
                        logger.warning("Action step missing tool name or parameters")
                        
                elif reasoning_step.step_type == ReasoningStepType.FINAL_ANSWER:
                    # Final answer reached
                    log_section_end("ReAct Reasoning")
                    memory.add_message(MessageRole.ASSISTANT, reasoning_step.content)
                    return reasoning_step.content
                
                step_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in reasoning step {step_count + 1}: {str(e)}", exc_info=True)
                
                # Add error step to memory
                error_step = ReasoningStep(
                    step_type=ReasoningStepType.OBSERVE,
                    content=f"Error occurred: {str(e)}. Let me try a different approach.",
                    metadata={"error": str(e), "step_count": step_count}
                )
                memory.add_reasoning_step(error_step)
                
                if error_count >= self.max_errors:
                    logger.error("Maximum errors reached, stopping reasoning")
                    break
        
        # If we reach here, we've exhausted steps or errors
        final_answer = "I apologize, but I was unable to solve this problem within the given constraints."
        if step_count >= self.max_reasoning_steps:
            final_answer += " I reached the maximum number of reasoning steps."
        if error_count >= self.max_errors:
            final_answer += " I encountered too many errors."
            
        memory.add_message(MessageRole.ASSISTANT, final_answer)
        return final_answer
    
    async def _generate_reasoning_step(self, memory: ReactMemory) -> ReasoningStep:
        """
        Generate the next reasoning step using LLM with structured responses.
        
        Args:
            memory: Agent memory
            
        Returns:
            Next reasoning step
        """
        # Build messages for LLM
        messages = self._build_llm_messages(memory)
        
        # Log the messages being sent to LLM
        prompt_text = "\n".join([f"Message {i+1} ({msg.role}):\n{msg.content}" for i, msg in enumerate(messages)])
        log_prompt("ReAct Engine", prompt_text, {"message_count": len(messages)})
        
        # Get structured LLM response
        if self.llm:
            action_response = await self.llm.generate_structured(messages, ReactAgentResponse)
        else:
            action_response = await self.llm_manager.generate_structured(messages, ReactAgentResponse)
        
        # Log the LLM response
        log_response("ReAct Engine", str(action_response))
        
        # Convert structured response to reasoning step
        return self._convert_action_to_reasoning_step(action_response)
    
    def _build_llm_messages(self, memory: ReactMemory) -> List[LLMMessage]:
        """
        Build messages for LLM from memory.
        
        Args:
            memory: Agent memory
            
        Returns:
            List of LLM messages
        """
        messages = []
        
        # System message (build dynamically to include current tools)
        system_prompt = self.custom_system_prompt or self._build_system_prompt()
        messages.append(LLMMessage("system", system_prompt))
        
        # Add conversation history
        for msg in memory.conversation_history:
            messages.append(LLMMessage(msg.role.value, msg.content))
        
        # Add recent reasoning context
        if memory.reasoning_trace:
            context_parts = []
            
            # Group reasoning steps for better context
            for step in memory.reasoning_trace[-10:]:  # Last 10 steps
                if step.step_type == ReasoningStepType.THINK:
                    context_parts.append(f"THINK: {step.content}")
                elif step.step_type == ReasoningStepType.ACT:
                    context_parts.append(f"ACT: {step.content}")
                elif step.step_type == ReasoningStepType.OBSERVE:
                    context_parts.append(f"OBSERVE: {step.content}")
            
            if context_parts:
                context_message = "Recent reasoning steps:\n" + "\n".join(context_parts)
                messages.append(LLMMessage("assistant", context_message))
        
        # Add prompt for next step
        messages.append(LLMMessage("user", "Continue with your reasoning. What is your next step?"))
        
        return messages
    
    def _format_tools_info(self) -> str:
        """
        Format available tools information for LLM.
        
        Returns:
            Formatted tools information
        """
        tools = self.tool_registry.get_all_tools()
        if not tools:
            return ""
        
        tool_descriptions = []
        tool_descriptions.append("AVAILABLE TOOLS:")
        
        for tool in tools:
            # Create detailed parameter descriptions
            params = []
            if tool.parameters:
                for param_name, param_info in tool.parameters.items():
                    if isinstance(param_info, dict):
                        param_type = param_info.get('type', 'str')
                        required = param_info.get('required', False)
                        param_desc = f"{param_name}: {param_type}"
                        if not required:
                            param_desc += " (optional)"
                        params.append(param_desc)
            
            param_str = f"({', '.join(params)})" if params else "()"
            tool_descriptions.append(f"- {tool.name}{param_str}: {tool.description}")
        
        return "\n".join(tool_descriptions)
    
    async def _execute_tool_action(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        memory: ReactMemory
    ) -> ToolResult:
        """
        Execute a tool action.
        
        Args:
            tool_name: Name of tool to execute
            tool_params: Parameters for tool
            memory: Agent memory
            
        Returns:
            Tool execution result
        """
        # Execute tool
        tool_result = await self.tool_registry.execute_tool(tool_name, **tool_params)
        
        # Log tool execution
        log_tool_execution(tool_name, tool_params, tool_result.result, tool_result.success)
        
        # Create tool execution record
        tool_execution = ToolExecution(
            tool_name=tool_name,
            params=tool_params,
            result=tool_result,
            timestamp=datetime.now()
        )
        memory.add_tool_execution(tool_execution)
        
        return tool_result
    
    async def _create_observation_step(
        self,
        tool_name: str,
        tool_result: ToolResult,
        memory: ReactMemory
    ) -> ReasoningStep:
        """
        Create an observation step based on tool result.
        
        Args:
            tool_name: Name of executed tool
            tool_result: Result of tool execution
            memory: Agent memory
            
        Returns:
            Observation reasoning step
        """
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
    
    def _convert_action_to_reasoning_step(self, action: ReactAgentResponse) -> ReasoningStep:
        """
        Convert structured action response to reasoning step.
        
        Args:
            action: Structured action from LLM
            
        Returns:
            Parsed reasoning step
        """
        try:
            if action.action == "think":
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=action.content or ""
                )
            
            elif action.action == "use_tool":
                if not action.tool_name:
                    raise ValueError("Tool name not specified in use_tool action")
                
                # Create action description for content
                param_strs = [f'{k}="{v}"' for k, v in (action.parameters or {}).items()]
                action_content = f"{action.tool_name}({', '.join(param_strs)})"
                
                return ReasoningStep(
                    step_type=ReasoningStepType.ACT,
                    content=action_content,
                    tool_name=action.tool_name,
                    tool_params=action.parameters or {}
                )
            
            elif action.action == "result":
                return ReasoningStep(
                    step_type=ReasoningStepType.FINAL_ANSWER,
                    content=action.content or ""
                )
            
            else:
                # Unknown action type, treat as thinking
                logger.warning(f"Unknown action type '{action.action}', treating as thinking")
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=f"Unknown action: {action.action}"
                )
                
        except Exception as e:
            logger.error(f"Failed to convert action to reasoning step: {e}, action: {action}")
            # Fallback to thinking step
            return ReasoningStep(
                step_type=ReasoningStepType.THINK,
                content=f"Failed to process action: {str(action)[:100]}"
            )
    
