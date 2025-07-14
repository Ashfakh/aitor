"""
Todo management system for task planning and execution.
Enables agents to break down complex problems into manageable tasks.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TodoPriority(Enum):
    """Priority levels for todo items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TodoItem:
    """Represents a single todo item in the task plan."""
    
    id: str
    content: str
    status: TodoStatus
    priority: TodoPriority
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
    
    def mark_completed(self, result: Optional[str] = None):
        """Mark todo as completed."""
        old_status = self.status
        self.status = TodoStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.result = result
        
    def mark_failed(self, error: str):
        """Mark todo as failed."""
        old_status = self.status
        self.status = TodoStatus.FAILED
        self.updated_at = datetime.now()
        self.error = error
        
    def mark_in_progress(self):
        """Mark todo as in progress."""
        old_status = self.status
        self.status = TodoStatus.IN_PROGRESS
        self.updated_at = datetime.now()
        
    def add_subtask(self, subtask_id: str):
        """Add a subtask to this todo."""
        if subtask_id not in self.subtasks:
            self.subtasks.append(subtask_id)
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert todo to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_id": self.parent_id,
            "subtasks": self.subtasks,
            "metadata": self.metadata,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TodoItem':
        """Create todo from dictionary representation."""
        return cls(
            id=data["id"],
            content=data["content"],
            status=TodoStatus(data["status"]),
            priority=TodoPriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
            parent_id=data.get("parent_id"),
            subtasks=data.get("subtasks", []),
            metadata=data.get("metadata", {}),
            result=data.get("result"),
            error=data.get("error")
        )


class TodoManager:
    """
    Manages todo items for task planning and execution.
    
    Provides capabilities for:
    - Creating and organizing todos
    - Tracking progress through task hierarchies
    - Executing todos with sub-agents
    - Managing dependencies between tasks
    """
    
    def __init__(self):
        self.todos: Dict[str, TodoItem] = {}
        self.execution_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def create_todo(
        self,
        content: str,
        priority: TodoPriority = TodoPriority.MEDIUM,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TodoItem:
        """Create a new todo item."""
        async with self._lock:
            todo = TodoItem(
                id=str(uuid4()),
                content=content,
                status=TodoStatus.PENDING,
                priority=priority,
                parent_id=parent_id,
                metadata=metadata or {}
            )
            
            self.todos[todo.id] = todo
            self.execution_order.append(todo.id)
            
            # Add to parent's subtasks if parent exists
            if parent_id and parent_id in self.todos:
                self.todos[parent_id].add_subtask(todo.id)
            
            logger.info(f"Created todo: {todo.id} - {content}")
            return todo
    
    async def get_todo(self, todo_id: str) -> Optional[TodoItem]:
        """Get todo by ID."""
        return self.todos.get(todo_id)
    
    async def get_todos_by_status(self, status: TodoStatus) -> List[TodoItem]:
        """Get all todos with specific status."""
        return [todo for todo in self.todos.values() if todo.status == status]
    
    async def get_next_pending_todo(self) -> Optional[TodoItem]:
        """Get the next pending todo to execute."""
        for todo_id in self.execution_order:
            todo = self.todos.get(todo_id)
            if todo and todo.status == TodoStatus.PENDING:
                # Check if all dependencies are completed
                if await self._are_dependencies_completed(todo):
                    return todo
        return None
    
    async def _are_dependencies_completed(self, todo: TodoItem) -> bool:
        """Check if all dependencies for a todo are completed."""
        # For now, we'll use a simple parent-child dependency model
        if todo.parent_id:
            parent = self.todos.get(todo.parent_id)
            if parent and parent.status not in [TodoStatus.COMPLETED, TodoStatus.FAILED]:
                return False
        return True
    
    async def mark_todo_completed(self, todo_id: str, result: Optional[str] = None):
        """Mark a todo as completed."""
        async with self._lock:
            todo = self.todos.get(todo_id)
            if todo:
                todo.mark_completed(result)
                logger.info(f"Completed todo: {todo_id} - {todo.content}")
    
    async def mark_todo_failed(self, todo_id: str, error: str):
        """Mark a todo as failed."""
        async with self._lock:
            todo = self.todos.get(todo_id)
            if todo:
                todo.mark_failed(error)
                logger.error(f"Failed todo: {todo_id} - {todo.content}: {error}")
    
    async def mark_todo_in_progress(self, todo_id: str):
        """Mark a todo as in progress."""
        async with self._lock:
            todo = self.todos.get(todo_id)
            if todo:
                todo.mark_in_progress()
                logger.info(f"Started todo: {todo_id} - {todo.content}")
    
    async def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of todo progress."""
        status_counts = {}
        for status in TodoStatus:
            status_counts[status.value] = len([
                todo for todo in self.todos.values() 
                if todo.status == status
            ])
        
        total_todos = len(self.todos)
        completed_todos = status_counts.get(TodoStatus.COMPLETED.value, 0)
        
        return {
            "total_todos": total_todos,
            "completed_todos": completed_todos,
            "progress_percentage": (completed_todos / total_todos * 100) if total_todos > 0 else 0,
            "status_breakdown": status_counts,
            "current_todo": await self.get_next_pending_todo()
        }
    
    async def create_plan_from_problem(self, problem: str, llm_generate_func: Callable) -> List[TodoItem]:
        """
        Create a todo plan from a problem description using LLM.
        
        Args:
            problem: Problem description
            llm_generate_func: Function to generate LLM responses
            
        Returns:
            List of created todo items
        """
        planning_prompt = f"""
        You are a task planning assistant. Break down the following problem into a clear, actionable todo list.
        
        Problem: {problem}
        
        Create a structured plan with specific, actionable tasks. Each task should be:
        1. Clear and specific
        2. Actionable (can be executed by tools or reasoning)
        3. Properly ordered (dependencies considered)
        4. Appropriately sized (not too big or too small)
        
        Format your response as a JSON array of tasks:
        [
            {{
                "content": "specific task description",
                "priority": "high|medium|low",
                "dependencies": ["task_index_that_must_complete_first"]
            }}
        ]
        
        Example:
        [
            {{
                "content": "Analyze the current system architecture",
                "priority": "high",
                "dependencies": []
            }},
            {{
                "content": "Identify performance bottlenecks",
                "priority": "medium", 
                "dependencies": ["0"]
            }}
        ]
        
        Respond with ONLY the JSON array, no additional text.
        """
        
        # Generate plan using LLM
        from .logging_config import log_prompt, log_response
        log_prompt("Todo Manager", planning_prompt, {"problem_length": len(problem)})
        
        response = await llm_generate_func([{"role": "user", "content": planning_prompt}])
        
        log_response("Todo Manager", response.content, {"response_length": len(response.content)})
        
        try:
            # Parse the JSON response
            plan_data = json.loads(response.content.strip())
            
            # Create todos from plan
            created_todos = []
            todo_mapping = {}  # Index to todo_id mapping
            
            for i, task_data in enumerate(plan_data):
                priority = TodoPriority(task_data.get("priority", "medium"))
                
                todo = await self.create_todo(
                    content=task_data["content"],
                    priority=priority,
                    metadata={"plan_index": i, "dependencies": task_data.get("dependencies", [])}
                )
                
                created_todos.append(todo)
                todo_mapping[i] = todo.id
            
            # Set up dependencies (for now, we'll implement this as execution order)
            # In a more sophisticated system, we'd track actual dependencies
            
            logger.info(f"Created {len(created_todos)} todos from problem: {problem}")
            return created_todos
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM planning response: {e}")
            # Fallback: create a single todo
            todo = await self.create_todo(
                content=f"Solve: {problem}",
                priority=TodoPriority.HIGH,
                metadata={"fallback": True}
            )
            return [todo]
    
    def get_all_todos(self) -> List[TodoItem]:
        """Get all todos in execution order."""
        return [self.todos[todo_id] for todo_id in self.execution_order if todo_id in self.todos]
    
    def get_todo_tree(self) -> Dict[str, Any]:
        """Get todos organized in a tree structure."""
        root_todos = [todo for todo in self.todos.values() if not todo.parent_id]
        
        def build_tree(todo: TodoItem) -> Dict[str, Any]:
            return {
                "todo": todo.to_dict(),
                "subtasks": [build_tree(self.todos[subtask_id]) for subtask_id in todo.subtasks if subtask_id in self.todos]
            }
        
        return {
            "root_todos": [build_tree(todo) for todo in root_todos],
            "total_todos": len(self.todos)
        }
    
    async def clear_todos(self):
        """Clear all todos."""
        async with self._lock:
            self.todos.clear()
            self.execution_order.clear()
            logger.info("Cleared all todos")