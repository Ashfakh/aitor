"""
Planning-based reasoning engine for intelligent agents.
Implements task planning, todo management, and sub-agent execution like Claude Code.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .react_agent import ReactAgent

from .llm import BaseLLM, LLMManager, Message as LLMMessage
from pydantic import BaseModel, Field
from typing import List as ListType
from .memory import ReactMemory, ReasoningStep, ReasoningStepType, MessageRole
from .tools import ToolRegistry, ToolResult, ToolExecution
from .todo import TodoManager, TodoItem, TodoStatus, TodoPriority
from .logging_config import (
    log_prompt, log_response, log_reasoning_step, log_tool_execution, 
    log_section_start, log_section_end, log_todo_created, log_todo_status_change, 
    log_sub_agent_execution, log_planning_summary
)

logger = logging.getLogger(__name__)


# Pydantic models for structured todo planning
class PlannedTodo(BaseModel):
    """Model for a single planned todo item."""
    content: str = Field(..., description="Clear, actionable task description")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, or urgent")


class TodoPlan(BaseModel):
    """Model for the complete todo plan."""
    todos: ListType[PlannedTodo] = Field(..., description="List of todo items to execute")


@dataclass
class ReactAgentResult:
    """Result of ReactAgent execution."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    tool_executions: List[ToolExecution] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        
        if not self.llm and not self.llm_manager:
            raise ValueError("Either llm or llm_manager must be provided")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for planning agent."""
        return """You are an advanced planning agent that efficiently solves problems by creating minimal, focused task breakdowns.

CRITICAL: You MUST respond with valid JSON only when requested. Use the structured format for tool usage.

Your approach:
1. PLAN: Analyze the problem complexity and create the minimal necessary todo list
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
- ONLY break down truly complex problems that require multiple distinct steps
- For simple tasks, create just ONE todo item that can be completed directly
- Avoid over-planning: prefer fewer, more comprehensive todos over many small ones
- Each todo should be substantial enough to warrant separate execution
- Use sub-agents for bounded, focused tasks
- Track progress and adapt plans as needed
- Provide clear, comprehensive final answers
- Be systematic but not overly granular in your approach"""
    
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
        
        # Generate plan using the planning agent's LLM with structured output
        planning_prompt = f"""Analyze the following problem and determine if it needs to be broken down into multiple tasks.

Problem: {problem}

Guidelines:
- For simple tasks that can be completed in one step, create just ONE todo
- Only break down truly complex problems that require multiple distinct steps
- Avoid over-planning: prefer fewer, more comprehensive todos over many small ones
- Each todo should be substantial enough to warrant separate execution

Create the minimal necessary todo list."""

        messages = [
            LLMMessage("system", "You are a planning agent that creates minimal, efficient task breakdowns."),
            LLMMessage("user", planning_prompt)
        ]
        
        # Get structured response
        if self.llm:
            todo_plan = await self.llm.generate_structured(messages, TodoPlan)
        else:
            todo_plan = await self.llm_manager.generate_structured(messages, TodoPlan)
        
        # Create todos from structured response
        todos = []
        for planned_todo in todo_plan.todos:
            todo = await self.todo_manager.create_todo(
                content=planned_todo.content,
                priority=TodoPriority(planned_todo.priority)
            )
            todos.append(todo)
        
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
    
    async def _execute_todo(self, todo: TodoItem, memory: ReactMemory) -> 'ReactAgentResult':
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
        
        # Execute with ReactAgent for bounded execution
        return await self._execute_with_react_agent(todo, memory)
    
    
    async def _execute_with_react_agent(self, todo: TodoItem, memory: ReactMemory) -> 'ReactAgentResult':
        """Execute todo using a ReactAgent."""
        # Create specialized agent based on todo type
        agent_name = self._get_agent_name(todo)
        agent_role = f"{agent_name} specialized in executing focused tasks efficiently"
        additional_instructions = self._get_agent_instructions(todo)
        
        # Create context for agent
        context = {
            "todo_id": todo.id,
            "priority": todo.priority.value,
            "parent_problem": self._get_parent_problem(memory)
        }
        
        # Log delegation
        log_reasoning_step(
            "Planning Engine",
            "DELEGATE",
            f"Delegating to ReactAgent '{agent_name}': {todo.content}",
            {"agent": agent_name, "todo_id": todo.id}
        )
        
        # Execute with ReactAgent (isolated memory per task)
        result = await self._execute_with_react_sub_agent(
            task=todo.content,
            agent_name=agent_name,
            context=context,
            agent_role=agent_role,
            additional_instructions=additional_instructions,
            max_reasoning_steps=10
        )
        
        # Log agent execution result
        log_sub_agent_execution(agent_name, todo.content, result.result, result.error)
        
        # Add delegation step to memory
        delegation_step = ReasoningStep(
            step_type=ReasoningStepType.THINK,
            content=f"Delegated to ReactAgent '{agent_name}': {todo.content}",
            metadata={
                "todo_id": todo.id,
                "agent": agent_name,
                "phase": "delegation"
            }
        )
        memory.add_reasoning_step(delegation_step)
        
        return result
    
    
    def _get_agent_name(self, todo: TodoItem) -> str:
        """Get appropriate agent name for todo."""
        content_lower = todo.content.lower()
        
        if any(word in content_lower for word in ["calculate", "compute", "math"]):
            return "calculator_agent"
        elif any(word in content_lower for word in ["search", "find", "lookup"]):
            return "search_agent"
        elif any(word in content_lower for word in ["analyze", "check", "verify"]):
            return "analysis_agent"
        else:
            return "general_agent"
    
    def _get_agent_instructions(self, todo: TodoItem) -> str:
        """Get additional task-specific instructions for agent based on todo type."""
        instructions = """- Focus solely on completing the assigned task
- Always use the available tools with their exact names and required parameters
- Provide clear, actionable results
- Be concise and direct
- When you have the result, use "result" action to complete the task"""
        
        return instructions
    
    def _get_agent_prompt(self, todo: TodoItem) -> str:
        """Get specialized prompt for agent based on todo type."""
        agent_name = self._get_agent_name(todo)
        
        # Get available tools from the tool registry
        available_tools = self.tool_registry.get_all_tools()
        tools_description = []
        
        for tool in available_tools:
            # Create tool description with detailed parameters
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
            tools_description.append(f"- {tool.name}{param_str}: {tool.description}")
        
        tools_list = "\n".join(tools_description) if tools_description else "- No tools available"
        
        base_prompt = f"""You are a {agent_name} specialized in executing focused tasks efficiently using ReAct reasoning.

Your goal: {todo.content}

Available tools:
{tools_list}

You operate in a loop of Thought, Action, and Observation. You must respond with structured JSON in one of these formats:

For thinking/reasoning:
{{"action": "think", "content": "Your reasoning about the problem and what to do next"}}

For using a tool:
{{"action": "use_tool", "tool_name": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}

For providing the final answer:
{{"action": "result", "content": "Your complete answer to the user's question"}}

Instructions:
- Focus solely on completing the assigned task
- Always use the available tools with their exact names and required parameters
- Always respond with valid JSON in the specified format
- Provide clear, actionable results
- Be concise and direct
- When you have the result, use "result" action to complete the task
"""
        
        return base_prompt
    
    async def _execute_with_react_sub_agent(
        self,
        task: str,
        agent_name: str,
        context: Dict[str, Any],
        agent_role: str,
        additional_instructions: str,
        max_reasoning_steps: int
    ) -> 'ReactAgentResult':
        """Execute task using a fresh ReactAgent instance (isolated memory)."""
        from .react_agent import ReactAgent
        import time
        
        start_time = time.time()
        
        try:
            # Create a new ReactAgent instance for this task (isolated memory)
            sub_agent = ReactAgent(
                name=f"{agent_name}_{task[:20]}",  # Unique name per task
                llm_manager=self.llm_manager,
                tool_registry=self.tool_registry,
                max_reasoning_steps=max_reasoning_steps,
                max_errors=3,
                task_goal=task,
                agent_role=agent_role,
                additional_instructions=additional_instructions
            )
            
            # Since task_goal is already in system prompt, just pass context if available
            # if context:
            #     context_message = f"Context: {context}"
            #     result = await self._execute_with_direct_tools(sub_agent, context_message)
            # else:
            #     # Task is already in system prompt, so pass a simple start message
            #     result = await self._execute_with_direct_tools(sub_agent, "Begin working on the task.")
            
            result = await self._execute_with_direct_tools(sub_agent, "Complete the task.")
            
            execution_time = time.time() - start_time
            
            # Return ReactAgentResult format for compatibility
            return ReactAgentResult(
                success=True,
                result=result,
                reasoning_steps=sub_agent.memory.reasoning_trace,
                tool_executions=sub_agent.memory.tool_executions,
                execution_time=execution_time,
                metadata={"agent_name": agent_name, "task": task}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ReactAgentResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_name": agent_name, "task": task}
            )
    
    async def _execute_with_direct_tools(self, sub_agent: ReactAgent, task: str) -> str:
        """Execute sub-agent with direct access to parent tools."""
        # Give sub-agent direct access to parent's tool registry
        sub_agent.tool_registry = self.tool_registry
        
        # Execute the task - sub-agent can now execute tools directly
        result = await sub_agent.solve(task)
        
        return result
    
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
        
        print(f"Final answer prompt=======: {final_answer_prompt}")
        
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
    
    
    async def get_planning_summary(self) -> Dict[str, Any]:
        """Get summary of planning state."""
        progress = await self.todo_manager.get_progress_summary()
        
        return {
            "todo_progress": progress,
            "todo_tree": self.todo_manager.get_todo_tree()
        }