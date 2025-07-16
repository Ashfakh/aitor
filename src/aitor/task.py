import inspect
from typing import Any, Callable, Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class Task:
    """Encapsulates a Python function as a task in a workflow."""
    
    def __init__(self, func: Callable, name: Optional[str] = None):
        """
        Initialize a task with a function.
        
        Args:
            func: The Python function to execute
            name: Optional name for the task (defaults to function name)
        """
        self.func = func
        self.name = name or func.__name__
        self.upstream_tasks: List['Task'] = []
        self.downstream_tasks: List['Task'] = []
        self.upstream_order: Dict['Task', int] = {}  # Maps task to position
        self.signature = inspect.signature(func)
        
    def __rshift__(self, other: Union['Task', List['Task']]) -> Union['Task', List['Task']]:
        """
        Implement the >> operator for task chaining.
        
        Examples:
            task1 >> task2
            task1 >> [task2, task3]
        """
        if isinstance(other, Task):
            self.downstream_tasks.append(other)
            other.upstream_tasks.append(self)
            other.upstream_order[self] = len(other.upstream_tasks) - 1
            return other
        elif isinstance(other, list):
            for task in other:
                if not isinstance(task, Task):
                    raise TypeError(f"Expected Task object, got {type(task)}")
                self.downstream_tasks.append(task)
                task.upstream_tasks.append(self)
                task.upstream_order[self] = len(task.upstream_tasks) - 1
            return other
        else:
            raise TypeError(f"Cannot chain with object of type {type(other)}")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the task's function with the given arguments."""
        logger.info(f"Executing task: {self.name}")
        return self.func(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"Task({self.name})"


def task(func=None, *, name=None):
    """Decorator to create a Task from a function."""
    def decorator(fn):
        return Task(fn, name=name)
    
    if func is None:
        return decorator
    return Task(func) 