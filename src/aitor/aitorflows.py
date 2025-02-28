import concurrent.futures
from typing import Any, Dict, List, Set, Optional
from collections import deque
import inspect
import logging
from aitor.task import Task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Aitorflow:
    """Manages a directed acyclic graph (DAG) of tasks."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a workflow.
        
        Args:
            name: Optional name for the workflow
        """
        self.name = name or "Aitorflow"
        self.tasks: Set[Task] = set()
        
    def add_task(self, task: Task) -> 'Aitorflow':
        """Add a task to the workflow and return self for chaining."""
        self.tasks.add(task)
        # Also add connected tasks
        queue = deque([task])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            self.tasks.add(current)
            
            # Add all connected tasks
            for upstream in current.upstream_tasks:
                if upstream not in visited:
                    queue.append(upstream)
            
            for downstream in current.downstream_tasks:
                if downstream not in visited:
                    queue.append(downstream)
                    
        return self
    
    def get_entry_tasks(self) -> List[Task]:
        """Return tasks with no upstream dependencies."""
        return [task for task in self.tasks if not task.upstream_tasks]
    
    def get_exit_tasks(self) -> List[Task]:
        """Return tasks with no downstream dependencies."""
        return [task for task in self.tasks if not task.downstream_tasks]
    
    def validate(self) -> bool:
        """
        Validate the workflow structure and input/output compatibility.
        
        Raises:
            ValueError: If validation fails
        
        Returns:
            True if validation succeeds
        """
        if not self.tasks:
            raise ValueError("Workflow has no tasks")
            
        # Check for cycles using DFS
        for task in self.tasks:
            visited = set()
            path = set()
            
            def check_cycle(current: Task) -> bool:
                visited.add(current)
                path.add(current)
                
                for downstream in current.downstream_tasks:
                    if downstream in path:
                        return True
                    if downstream not in visited:
                        if check_cycle(downstream):
                            return True
                
                path.remove(current)
                return False
            
            if task not in visited:
                if check_cycle(task):
                    raise ValueError(f"Cycle detected in workflow involving task: {task.name}")
        
        # Validate input/output compatibility
        for task in self.tasks:
            if task.upstream_tasks:
                # For tasks with multiple inputs, we need to ensure they can accept
                # either individual arguments or a tuple of arguments
                parameters = list(task.signature.parameters.values())
                if len(task.upstream_tasks) > 1:
                    # If multiple upstream tasks, check if this task can accept
                    # either multiple positional args or a tuple
                    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
                    can_accept_single_tuple = (len(parameters) == 1 and
                                              parameters[0].annotation != inspect.Signature.empty and
                                              (parameters[0].annotation == tuple or 
                                               getattr(parameters[0].annotation, '__origin__', None) == tuple))
                    
                    if not (has_var_positional or can_accept_single_tuple):
                        logger.warning(
                            f"Task {task.name} has {len(task.upstream_tasks)} upstream tasks "
                            f"but doesn't have *args parameter or accept a single tuple"
                        )
        
        return True
    
    def execute(self, initial_input: Any = None) -> Dict[str, Any]:
        """
        Execute the workflow with the given initial input.
        
        Args:
            initial_input: Input to pass to entry tasks
            
        Returns:
            Dictionary mapping task names to their results
        """
        self.validate()
        
        # Track task completion and results
        results: Dict[Task, Any] = {}
        completed_tasks: Set[Task] = set()
        
        # Create a copy of the task dependencies to track remaining dependencies
        remaining_deps: Dict[Task, Set[Task]] = {
            task: set(task.upstream_tasks) for task in self.tasks
        }
        
        # Start with entry tasks
        ready_tasks = self.get_entry_tasks()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures: Dict[concurrent.futures.Future, Task] = {}
            
            # Submit entry tasks
            for task in ready_tasks:
                if task.upstream_tasks:
                    raise ValueError(f"Entry task {task.name} has upstream dependencies")
                
                if initial_input is not None:
                    futures[executor.submit(task.execute, initial_input)] = task
                else:
                    futures[executor.submit(task.execute)] = task
            
            # Process tasks as they complete
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    task = futures.pop(future)
                    try:
                        result = future.result()
                        logger.info(f"Task {task.name} completed successfully")
                        results[task] = result
                        completed_tasks.add(task)
                        
                        # Check for downstream tasks that are ready to execute
                        for downstream in task.downstream_tasks:
                            remaining_deps[downstream].discard(task)
                            
                            if not remaining_deps[downstream]:
                                # All upstream dependencies are completed
                                # Gather inputs from all upstream tasks
                                upstream_results = [results[up] for up in downstream.upstream_tasks]
                                
                                if len(upstream_results) == 1:
                                    # Single input
                                    futures[executor.submit(downstream.execute, upstream_results[0])] = downstream
                                else:
                                    # Multiple inputs as tuple
                                    futures[executor.submit(downstream.execute, *upstream_results)] = downstream
                    
                    except Exception as e:
                        logger.error(f"Task {task.name} failed with error: {str(e)}")
                        raise RuntimeError(f"Task {task.name} failed: {str(e)}") from e
        
        # Return results for all tasks
        return {task.name: results[task] for task in self.tasks if task in results}

    
    def print(self) -> None:
        """
        Print an ASCII diagram of the workflow to the terminal.
        Shows tasks and their dependencies in a visual format.
        """
        if not self.tasks:
            print(f"Workflow '{self.name}' is empty")
            return
            
        print(f"\n=== Workflow: {self.name} ===\n")
        
        # Assign levels to tasks (longest path from entry)
        levels = {}
        
        def assign_level(task, level=0):
            if task in levels:
                # If already assigned, take the maximum level
                levels[task] = max(levels[task], level)
            else:
                levels[task] = level
                
            for downstream in task.downstream_tasks:
                assign_level(downstream, level + 1)
        
        # Start with entry tasks
        for task in self.get_entry_tasks():
            assign_level(task)
        
        # Group tasks by level
        tasks_by_level = {}
        for task, level in levels.items():
            if level not in tasks_by_level:
                tasks_by_level[level] = []
            tasks_by_level[level].append(task)
        
        # Print level by level
        max_level = max(levels.values()) if levels else 0
        
        for level in range(max_level + 1):
            if level not in tasks_by_level:
                continue
            
            # Print level header
            print(f"Level {level}:")
            
            # Sort tasks within level by name for consistent output
            tasks = sorted(tasks_by_level[level], key=lambda t: t.name)
            
            for task in tasks:
                print(f"  {task.name}")
                
                # Print connections
                if task.downstream_tasks:
                    connections = sorted([f"{t.name} (L{levels[t]})" for t in task.downstream_tasks], 
                                        key=lambda s: s.split()[0])  # Sort by task name
                    connections_str = ", ".join(connections)
                    print(f"    → {connections_str}")
            
            print()  # Empty line between levels
        
        print(f"Total tasks: {len(self.tasks)}") 