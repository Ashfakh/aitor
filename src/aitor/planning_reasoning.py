"""
Planning-based reasoning engine for intelligent agents.
Implements task planning, todo management, and sub-agent execution like Claude Code.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from .llm import BaseLLM, LLMManager, Message as LLMMessage
from .memory import ReactMemory, ReasoningStep, ReasoningStepType, MessageRole
from .tools import ToolRegistry, ToolResult, ToolExecution
from .todo import TodoManager, TodoItem, TodoStatus, TodoPriority
from .sub_agent import SubAgentManager, SubAgentResult
from .logging_config import (
    log_prompt, log_response, log_reasoning_step, log_tool_execution, 
    log_section_start, log_section_end, log_todo_created, log_todo_status_change, 
    log_sub_agent_execution, log_planning_summary
)

logger = logging.getLogger(__name__)


class PlanningReasoningEngine:
    """
    Advanced reasoning engine that uses planning and sub-agents.
    
    Features:
    - Task planning and todo management
    - Sub-agent execution for bounded tasks
    - Progress tracking and reporting
    - Adaptive planning based on results
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: Optional[BaseLLM] = None,
        llm_manager: Optional[LLMManager] = None,
        max_reasoning_steps: int = 50,
        max_errors: int = 5,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize planning reasoning engine.
        
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
        
        # Planning components
        self.todo_manager = TodoManager()
        self.sub_agent_manager = SubAgentManager(tool_registry, llm, llm_manager)
        
        if not self.llm and not self.llm_manager:
            raise ValueError("Either llm or llm_manager must be provided")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for planning agent."""
        return """You are an advanced planning agent that breaks down complex problems into manageable tasks and executes them systematically.

CRITICAL: You MUST respond with valid JSON only when requested. Use the structured format for tool usage.

Your approach:
1. PLAN: Analyze the problem and create a structured todo list
2. EXECUTE: Work through todos one by one using tools or sub-agents
3. ADAPT: Adjust the plan based on results and new information
4. COMPLETE: Provide comprehensive final answers

For tool usage, respond with JSON:
{
    "action": "use_tool",
    "tool_name": "tool_name",
    "parameters": {
        "param1": "value1"
    }
}

For other responses, use standard text format:
- PLAN: [When creating or updating the task plan]
- EXECUTE: [When working on a specific todo]
- THINK: [When analyzing or reasoning about the current state]
- DELEGATE: [When assigning a task to a sub-agent]
- COMPLETE: [When the entire problem is solved]

Key principles:
- Break complex problems into specific, actionable tasks
- Use sub-agents for bounded, focused tasks
- Track progress and adapt plans as needed
- Provide clear, comprehensive final answers
- Be systematic and thorough in your approach"""
    
    async def solve_problem(self, problem: str, memory: ReactMemory) -> str:
        """
        Solve a problem using planning-based reasoning.
        
        Args:
            problem: Problem description
            memory: Agent memory
            
        Returns:
            Final answer or solution
        """
        log_section_start(f"Planning-based Reasoning: {problem[:50]}...")
        
        # Add problem to memory
        memory.add_message(MessageRole.USER, problem)
        
        # Phase 1: Create initial plan
        await self._create_initial_plan(problem, memory)
        
        # Phase 2: Execute plan
        final_answer = await self._execute_plan(memory)
        
        # Add final answer to memory
        memory.add_message(MessageRole.ASSISTANT, final_answer)
        
        log_section_end("Planning-based Reasoning")
        return final_answer
    
    async def _create_initial_plan(self, problem: str, memory: ReactMemory):
        """Create initial todo plan for the problem."""
        log_section_start("Creating Initial Plan")
        
        # Generate plan using LLM
        llm_generate_func = self._get_llm_generate_func()
        todos = await self.todo_manager.create_plan_from_problem(problem, llm_generate_func)
        
        # Log todos created
        for todo in todos:
            log_todo_created(todo.id, todo.content, todo.priority.value)
        
        # Add planning step to memory
        plan_summary = "\n".join([f"- {todo.content}" for todo in todos])
        planning_step = ReasoningStep(
            step_type=ReasoningStepType.THINK,
            content=f"Created execution plan:\n{plan_summary}",
            metadata={"phase": "planning", "todos_created": len(todos)}
        )
        memory.add_reasoning_step(planning_step)
        
        # Log planning step
        log_reasoning_step(
            "Planning Engine",
            "PLAN",
            f"Created execution plan with {len(todos)} todos:\n{plan_summary}",
            {"todos_created": len(todos)}
        )
        
        log_section_end("Creating Initial Plan")
    
    async def _execute_plan(self, memory: ReactMemory) -> str:
        """Execute the todo plan systematically."""
        log_section_start("Executing Plan")
        
        error_count = 0
        step_count = 0
        
        while step_count < self.max_reasoning_steps and error_count < self.max_errors:
            try:
                # Get next todo to execute
                next_todo = await self.todo_manager.get_next_pending_todo()
                
                if not next_todo:
                    # No more todos - check if we're done
                    progress = await self.todo_manager.get_progress_summary()
                    
                    # Log current progress
                    log_planning_summary(
                        progress["total_todos"],
                        progress["completed_todos"],
                        len(await self.todo_manager.get_todos_by_status(TodoStatus.FAILED)),
                        progress["total_todos"] - progress["completed_todos"]
                    )
                    
                    if progress["completed_todos"] == progress["total_todos"]:
                        # All todos completed - generate final answer
                        return await self._generate_final_answer(memory)
                    else:
                        # No pending todos but not all completed - may need new todos
                        await self._assess_and_adapt_plan(memory)
                        continue
                
                # Execute the todo
                result = await self._execute_todo(next_todo, memory)
                
                if result.success:
                    await self.todo_manager.mark_todo_completed(next_todo.id, result.result)
                    log_todo_status_change(next_todo.id, next_todo.content, "in_progress", "completed")
                else:
                    await self.todo_manager.mark_todo_failed(next_todo.id, result.error)
                    log_todo_status_change(next_todo.id, next_todo.content, "in_progress", "failed")
                    error_count += 1
                
                step_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error in planning step {step_count + 1}: {str(e)}", exc_info=True)
                
                if error_count >= self.max_errors:
                    break
        
        # If we reach here, generate final answer based on current state
        return await self._generate_final_answer(memory)
    
    async def _execute_todo(self, todo: TodoItem, memory: ReactMemory) -> SubAgentResult:
        """Execute a specific todo item."""
        # Mark todo as in progress
        await self.todo_manager.mark_todo_in_progress(todo.id)
        log_todo_status_change(todo.id, todo.content, "pending", "in_progress")
        
        # Add execution step to memory
        execution_step = ReasoningStep(
            step_type=ReasoningStepType.THINK,
            content=f"Executing todo: {todo.content}",
            metadata={"todo_id": todo.id, "phase": "execution"}
        )
        memory.add_reasoning_step(execution_step)
        
        # Log execution step
        log_reasoning_step(
            "Planning Engine",
            "EXECUTE",
            f"Executing todo: {todo.content}",
            {"todo_id": todo.id, "priority": todo.priority.value}
        )
        
        # Determine execution strategy
        if await self._should_delegate_to_sub_agent(todo):
            # Use sub-agent for bounded execution
            return await self._execute_with_sub_agent(todo, memory)
        else:
            # Execute directly with main agent
            return await self._execute_directly(todo, memory)
    
    async def _should_delegate_to_sub_agent(self, todo: TodoItem) -> bool:
        """Determine if todo should be delegated to sub-agent."""
        # Delegate if:
        # 1. Todo is a specific, bounded task
        # 2. Todo involves tool usage
        # 3. Todo doesn't require main context
        
        # For now, use simple heuristics
        content_lower = todo.content.lower()
        
        # Keywords that suggest sub-agent suitability
        sub_agent_keywords = [
            "calculate", "compute", "search", "find", "get", "fetch",
            "analyze", "check", "verify", "validate", "test",
            "convert", "transform", "format", "parse"
        ]
        
        return any(keyword in content_lower for keyword in sub_agent_keywords)
    
    async def _execute_with_sub_agent(self, todo: TodoItem, memory: ReactMemory) -> SubAgentResult:
        """Execute todo using a sub-agent."""
        # Create specialized sub-agent based on todo type
        agent_name = self._get_sub_agent_name(todo)
        system_prompt = self._get_sub_agent_prompt(todo)
        
        # Create context for sub-agent
        context = {
            "todo_id": todo.id,
            "priority": todo.priority.value,
            "parent_problem": self._get_parent_problem(memory)
        }
        
        # Log delegation
        log_reasoning_step(
            "Planning Engine",
            "DELEGATE",
            f"Delegating to sub-agent '{agent_name}': {todo.content}",
            {"sub_agent": agent_name, "todo_id": todo.id}
        )
        
        # Execute with sub-agent
        result = await self.sub_agent_manager.execute_with_sub_agent(
            task=todo.content,
            agent_name=agent_name,
            context=context,
            system_prompt=system_prompt,
            max_reasoning_steps=5,  # Reduced for efficiency
            timeout=30.0  # Reduced timeout
        )
        
        # Log sub-agent execution result
        log_sub_agent_execution(agent_name, todo.content, result.result, result.error)
        
        # Add delegation step to memory
        delegation_step = ReasoningStep(
            step_type=ReasoningStepType.THINK,
            content=f"Delegated to sub-agent '{agent_name}': {todo.content}",
            metadata={
                "todo_id": todo.id,
                "sub_agent": agent_name,
                "phase": "delegation"
            }
        )
        memory.add_reasoning_step(delegation_step)
        
        return result
    
    async def _execute_directly(self, todo: TodoItem, memory: ReactMemory) -> SubAgentResult:
        """Execute todo directly with main agent reasoning."""
        logger.info(f"Executing directly: {todo.content}")
        
        # Use main agent to execute the todo
        # This is a simplified implementation - in practice, you'd use more sophisticated reasoning
        
        try:
            # Generate reasoning for the todo
            reasoning_step = await self._generate_reasoning_step_for_todo(todo, memory)
            memory.add_reasoning_step(reasoning_step)
            
            # If it's an action, execute it
            if reasoning_step.step_type == ReasoningStepType.ACT:
                if reasoning_step.tool_name and reasoning_step.tool_params:
                    tool_result = await self._execute_tool_action(
                        reasoning_step.tool_name,
                        reasoning_step.tool_params,
                        memory
                    )
                    
                    # Create success result
                    return SubAgentResult(
                        success=tool_result.success,
                        result=str(tool_result.result) if tool_result.success else None,
                        error=tool_result.error if not tool_result.success else None,
                        reasoning_steps=[reasoning_step],
                        tool_executions=[ToolExecution(
                            tool_name=reasoning_step.tool_name,
                            params=reasoning_step.tool_params,
                            result=tool_result,
                            timestamp=datetime.now()
                        )]
                    )
            
            # For thinking steps, return the reasoning as result
            return SubAgentResult(
                success=True,
                result=reasoning_step.content,
                reasoning_steps=[reasoning_step]
            )
            
        except Exception as e:
            logger.error(f"Direct execution failed: {str(e)}")
            return SubAgentResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_reasoning_step_for_todo(self, todo: TodoItem, memory: ReactMemory) -> ReasoningStep:
        """Generate reasoning step for a specific todo."""
        # Build prompt for todo execution
        messages = self._build_todo_execution_messages(todo, memory)
        
        # Get LLM response
        if self.llm:
            llm_response = await self.llm.generate(messages)
        else:
            llm_response = await self.llm_manager.generate(messages)
        
        # Parse response
        return self._parse_reasoning_response(llm_response.content)
    
    def _build_todo_execution_messages(self, todo: TodoItem, memory: ReactMemory) -> List[LLMMessage]:
        """Build messages for todo execution."""
        messages = []
        
        # System message
        system_content = f"""You are executing a specific todo item as part of a larger plan.

Todo: {todo.content}
Priority: {todo.priority.value}

Focus on completing this specific task. Use tools when needed.
Respond with one of:
- THINK: [reasoning about the todo]
- ACT: tool_name(param="value") [to use a tool]
- COMPLETE: [when todo is finished]
"""
        
        # Add tools info
        tools_info = self._format_tools_info()
        if tools_info:
            system_content += "\n\n" + tools_info
        
        messages.append(LLMMessage("system", system_content))
        
        # Add recent context
        recent_steps = memory.reasoning_trace[-5:] if memory.reasoning_trace else []
        if recent_steps:
            context_parts = []
            for step in recent_steps:
                if step.step_type == ReasoningStepType.THINK:
                    context_parts.append(f"THINK: {step.content}")
                elif step.step_type == ReasoningStepType.ACT:
                    context_parts.append(f"ACT: {step.content}")
            
            if context_parts:
                context_message = "Recent context:\n" + "\n".join(context_parts)
                messages.append(LLMMessage("assistant", context_message))
        
        # Prompt for todo execution
        messages.append(LLMMessage("user", f"Execute this todo: {todo.content}"))
        
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
    
    async def _execute_tool_action(self, tool_name: str, tool_params: Dict[str, Any], memory: ReactMemory) -> ToolResult:
        """Execute a tool action."""
        logger.info(f"Executing tool: {tool_name} with params: {tool_params}")
        
        # Execute tool
        tool_result = await self.tool_registry.execute_tool(tool_name, **tool_params)
        
        # Create tool execution record
        tool_execution = ToolExecution(
            tool_name=tool_name,
            params=tool_params,
            result=tool_result,
            timestamp=datetime.now()
        )
        memory.add_tool_execution(tool_execution)
        
        return tool_result
    
    def _parse_reasoning_response(self, response: str) -> ReasoningStep:
        """Parse LLM response into reasoning step."""
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
                # Parse action call
                import re
                match = re.match(r'(\w+)\((.*?)\)$', action_text.strip())
                if match:
                    tool_name = match.group(1)
                    params_str = match.group(2).strip()
                    
                    # Parse parameters
                    parameters = {}
                    if params_str:
                        # Handle param="value" format
                        for param_match in re.finditer(r'(\w+)="([^"]*)"', params_str):
                            key = param_match.group(1)
                            value = param_match.group(2)
                            parameters[key] = value
                    
                    return ReasoningStep(
                        step_type=ReasoningStepType.ACT,
                        content=action_text,
                        tool_name=tool_name,
                        tool_params=parameters
                    )
                else:
                    raise ValueError(f"Invalid action format: {action_text}")
                    
            except Exception as e:
                logger.error(f"Failed to parse action: {e}")
                return ReasoningStep(
                    step_type=ReasoningStepType.THINK,
                    content=f"I tried to use a tool but the format was incorrect: {action_text}"
                )
        
        elif response.upper().startswith("COMPLETE:"):
            content = response[9:].strip()
            return ReasoningStep(
                step_type=ReasoningStepType.FINAL_ANSWER,
                content=content
            )
        
        elif "FINAL_ANSWER:" in response.upper():
            final_answer_index = response.upper().find("FINAL_ANSWER:")
            content = response[final_answer_index + 13:].strip()
            return ReasoningStep(
                step_type=ReasoningStepType.FINAL_ANSWER,
                content=content
            )
        
        else:
            # Default to thinking
            return ReasoningStep(
                step_type=ReasoningStepType.THINK,
                content=response
            )
    
    def _get_llm_generate_func(self) -> Callable:
        """Get LLM generation function."""
        async def generate_func(messages):
            llm_messages = [LLMMessage(msg["role"], msg["content"]) for msg in messages]
            if self.llm:
                return await self.llm.generate(llm_messages)
            else:
                return await self.llm_manager.generate(llm_messages)
        
        return generate_func
    
    def _get_sub_agent_name(self, todo: TodoItem) -> str:
        """Get appropriate sub-agent name for todo."""
        content_lower = todo.content.lower()
        
        if any(word in content_lower for word in ["calculate", "compute", "math"]):
            return "calculator_agent"
        elif any(word in content_lower for word in ["search", "find", "lookup"]):
            return "search_agent"
        elif any(word in content_lower for word in ["analyze", "check", "verify"]):
            return "analysis_agent"
        else:
            return "general_agent"
    
    def _get_sub_agent_prompt(self, todo: TodoItem) -> str:
        """Get specialized prompt for sub-agent based on todo type."""
        agent_name = self._get_sub_agent_name(todo)
        
        base_prompt = f"""You are a {agent_name} specialized in executing focused tasks efficiently.

Your goal: {todo.content}

Instructions:
- Focus solely on completing the assigned task
- Use available tools when appropriate
- Provide clear, actionable results
- Be concise and direct
- Use RESULT: [your result] when finished
"""
        
        return base_prompt
    
    def _get_parent_problem(self, memory: ReactMemory) -> str:
        """Get the original problem from memory."""
        for msg in memory.conversation_history:
            if msg.role == MessageRole.USER:
                return msg.content
        return "Unknown problem"
    
    async def _assess_and_adapt_plan(self, memory: ReactMemory):
        """Assess current progress and adapt plan if needed."""
        logger.info("Assessing and adapting plan...")
        
        # Get current progress
        progress = await self.todo_manager.get_progress_summary()
        
        # Check if we need to create new todos
        failed_todos = await self.todo_manager.get_todos_by_status(TodoStatus.FAILED)
        
        if failed_todos:
            # Create recovery todos for failed ones
            for failed_todo in failed_todos:
                recovery_content = f"Recover from failed task: {failed_todo.content}"
                await self.todo_manager.create_todo(
                    content=recovery_content,
                    priority=TodoPriority.HIGH,
                    metadata={"recovery_for": failed_todo.id}
                )
        
        # Add adaptation step to memory
        adaptation_step = ReasoningStep(
            step_type=ReasoningStepType.THINK,
            content=f"Assessed progress: {progress['completed_todos']}/{progress['total_todos']} todos completed. Adapted plan as needed.",
            metadata={"phase": "adaptation", "progress": progress}
        )
        memory.add_reasoning_step(adaptation_step)
    
    async def _generate_final_answer(self, memory: ReactMemory) -> str:
        """Generate final answer based on completed todos and results."""
        logger.info("Generating final answer...")
        
        # Get all completed todos and their results
        completed_todos = await self.todo_manager.get_todos_by_status(TodoStatus.COMPLETED)
        
        # Build summary of work done
        work_summary = []
        for todo in completed_todos:
            result_text = todo.result if todo.result else "Completed"
            work_summary.append(f"✓ {todo.content}: {result_text}")
        
        # Get failed todos
        failed_todos = await self.todo_manager.get_todos_by_status(TodoStatus.FAILED)
        
        # Create final answer prompt
        final_answer_prompt = f"""Based on the work completed, provide a comprehensive final answer to the original problem.

Work completed:
{chr(10).join(work_summary)}

{"Failed tasks:" + chr(10).join([f"✗ {todo.content}: {todo.error}" for todo in failed_todos]) if failed_todos else ""}

Original problem: {self._get_parent_problem(memory)}

Provide a clear, comprehensive final answer that addresses the original problem based on the work completed.
"""
        
        # Generate final answer
        messages = [LLMMessage("user", final_answer_prompt)]
        
        if self.llm:
            response = await self.llm.generate(messages)
        else:
            response = await self.llm_manager.generate(messages)
        
        final_answer = response.content.strip()
        
        # Add final answer step to memory
        final_step = ReasoningStep(
            step_type=ReasoningStepType.FINAL_ANSWER,
            content=final_answer,
            metadata={"phase": "completion", "todos_completed": len(completed_todos)}
        )
        memory.add_reasoning_step(final_step)
        
        return final_answer
    
    # Additional utility methods
    def get_todo_manager(self) -> TodoManager:
        """Get the todo manager instance."""
        return self.todo_manager
    
    def get_sub_agent_manager(self) -> SubAgentManager:
        """Get the sub-agent manager instance."""
        return self.sub_agent_manager
    
    async def get_planning_summary(self) -> Dict[str, Any]:
        """Get summary of planning state."""
        progress = await self.todo_manager.get_progress_summary()
        sub_agents = self.sub_agent_manager.list_sub_agents()
        
        return {
            "todo_progress": progress,
            "sub_agents": sub_agents,
            "todo_tree": self.todo_manager.get_todo_tree()
        }