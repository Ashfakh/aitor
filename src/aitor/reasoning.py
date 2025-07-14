"""
ReAct reasoning engine for intelligent agents.
Implements the Think -> Act -> Observe reasoning loop.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .llm import BaseLLM, LLMManager, Message as LLMMessage
from .memory import ReactMemory, ReasoningStep, ReasoningStepType, MessageRole
from .tools import ToolRegistry, ToolResult, ToolExecution
from .logging_config import log_prompt, log_response, log_reasoning_step, log_tool_execution, log_section_start, log_section_end

logger = logging.getLogger(__name__)


@dataclass
class ActionCall:
    """Represents a parsed action call from reasoning text."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    
    def __post_init__(self):
        if not self.tool_name:
            raise ValueError("Tool name cannot be empty")


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
        system_prompt: Optional[str] = None
    ):
        """
        Initialize reasoning engine.
        
        Args:
            tool_registry: Registry of available tools
            llm: Direct LLM instance to use
            llm_manager: LLM manager for multiple providers
            max_reasoning_steps: Maximum reasoning steps per problem
            max_errors: Maximum errors before stopping
            system_prompt: Custom system prompt for the agent
        """
        self.tool_registry = tool_registry
        self.llm = llm
        self.llm_manager = llm_manager
        self.max_reasoning_steps = max_reasoning_steps
        self.max_errors = max_errors
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        if not self.llm and not self.llm_manager:
            raise ValueError("Either llm or llm_manager must be provided")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for ReAct agent."""
        return """You are a ReAct (Reasoning and Acting) agent. Your task is to solve problems by thinking step by step and using available tools when needed.

You operate in a loop of Thought, Action, and Observation:
1. Thought: Analyze the problem and plan your approach
2. Action: Use tools to gather information or perform tasks
3. Observation: Analyze the results and continue reasoning

Always format your responses in one of these ways:

For thinking:
THINK: [Your reasoning about the problem and what to do next]

For using a tool:
ACT: tool_name(param1="value1", param2="value2")

For providing the final answer:
FINAL_ANSWER: [Your complete answer to the user's question]

Important:
- Be thorough in your thinking before taking actions
- Use tools when you need specific information or to perform tasks
- Analyze observations carefully before proceeding
- Provide a clear, complete final answer when you have enough information
- If you encounter errors, adapt your approach and try alternatives"""
    
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
        Generate the next reasoning step using LLM.
        
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
        
        # Get LLM response
        if self.llm:
            llm_response = await self.llm.generate(messages)
        else:
            llm_response = await self.llm_manager.generate(messages)
        
        # Log the LLM response
        log_response("ReAct Engine", llm_response.content)
        
        # Parse response to determine step type and content
        return await self._parse_reasoning_response(llm_response.content)
    
    def _build_llm_messages(self, memory: ReactMemory) -> List[LLMMessage]:
        """
        Build messages for LLM from memory.
        
        Args:
            memory: Agent memory
            
        Returns:
            List of LLM messages
        """
        messages = []
        
        # System message
        system_content = self.system_prompt
        
        # Add available tools to system message
        tools_info = self._format_tools_info()
        if tools_info:
            system_content += "\n\n" + tools_info
        
        messages.append(LLMMessage("system", system_content))
        
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
            # Format parameters
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
    
    async def _parse_reasoning_response(self, response: str) -> ReasoningStep:
        """
        Parse LLM response into a reasoning step.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed reasoning step
        """
        response = response.strip()
        
        # Parse different step types
        if response.upper().startswith("THINK:"):
            content = response[6:].strip()
            return ReasoningStep(
                step_type=ReasoningStepType.THINK,
                content=content
            )
            
        elif response.upper().startswith("ACT:"):
            action_text = response[4:].strip()
            try:
                action_call = await self._parse_action_call(action_text)
                return ReasoningStep(
                    step_type=ReasoningStepType.ACT,
                    content=action_text,
                    tool_name=action_call.tool_name,
                    tool_params=action_call.parameters
                )
            except Exception as e:
                logger.error(f"Failed to parse action: {e}")
                # Return as thinking step if parsing fails
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=f"I tried to use a tool but the format was incorrect: {action_text}"
                )
                
        elif "FINAL_ANSWER:" in response.upper():
            # Find the FINAL_ANSWER anywhere in the response
            final_answer_index = response.upper().find("FINAL_ANSWER:")
            content = response[final_answer_index + 13:].strip()
            return ReasoningStep(
                step_type=ReasoningStepType.FINAL_ANSWER,
                content=content
            )
            
        else:
            # Default to thinking step if format is not recognized
            return ReasoningStep(
                step_type=ReasoningStepType.THINK,
                content=response
            )
    
    async def _parse_action_call(self, action_text: str) -> ActionCall:
        """
        Parse action call from text.
        
        Args:
            action_text: Text containing action call
            
        Returns:
            Parsed action call
        """
        # Try to parse as function call format: tool_name(param1="value1", param2=value2)
        match = re.match(r'(\w+)\((.*?)\)$', action_text.strip())
        if not match:
            raise ValueError(f"Invalid action format: {action_text}")
        
        tool_name = match.group(1)
        params_str = match.group(2).strip()
        
        # Parse parameters
        parameters = {}
        
        if params_str:
            # Try to parse as Python-like arguments
            try:
                # Build a safe dict for evaluation
                safe_dict = {"true": True, "false": False, "null": None}
                
                # Replace parameter format to be JSON-compatible
                # Handle both param="value" and param=value formats
                params_str = re.sub(r'(\w+)=', r'"\1": ', params_str)
                
                # Wrap in braces to make it a JSON object
                json_str = "{" + params_str + "}"
                
                # Parse as JSON
                parameters = json.loads(json_str)
                
            except Exception as e:
                logger.error(f"Failed to parse parameters as JSON: {e}")
                
                # Fallback to simple parsing
                for param_match in re.finditer(r'(\w+)=([^,]+)', params_str):
                    key = param_match.group(1)
                    value = param_match.group(2).strip()
                    
                    # Clean up value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.lower() == 'none' or value.lower() == 'null':
                        value = None
                    else:
                        # Try to parse as number
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string
                    
                    parameters[key] = value
        
        return ActionCall(
            tool_name=tool_name,
            parameters=parameters,
            reasoning=action_text
        )