# Aitor

Aitor is a Python framework for building intelligent, memory-enabled agents (aitors) that can execute complex workflows. It provides a robust foundation for creating asynchronous, task-based applications with a focus on clean architecture and developer experience.

## Features
- **Memory-Enabled Agents**: Create stateful agents that maintain memory between interactions.
- **Directed Acyclic Graph (DAG) Workflows**: Design complex task dependencies with simple syntax.
- **Async Support**: Built-in asynchronous processing with both blocking and non-blocking APIs.
- **Thread Safety**: Secure memory access across concurrent operations.
- **Visualization**: Built-in workflow visualization tools.

## Installation
```sh
pip install aitor
```

## Quick Start
### Creating a Basic Aitor
```python
from aitor import Aitor, Memory

class MyMemory(Memory):
    value: int = 0

class MyAitor(Aitor[MyMemory]):
    async def process(self, input_value: int):
        self.memory.value += input_value
        return self.memory.value

aitor = MyAitor()
result = aitor.process(10)
print(result)  # Output: 10
```

### Building Workflows
```python
from aitor import Task, Workflow

def task1():
    return "Hello"

def task2(msg: str):
    return f"{msg}, World!"

t1 = Task(task1)
t2 = Task(task2, t1)
workflow = Workflow(t1, t2)

result = workflow.run()
print(result)  # Output: "Hello, World!"
```

### Attaching Workflows to Aitors
```python
class WorkflowAitor(Aitor[MyMemory]):
    async def process(self):
        return await workflow.run_async()

workflow_aitor = WorkflowAitor()
result = workflow_aitor.process()
print(result)
```

## Core Concepts
### Aitors
Aitors are memory-enabled agents that can process messages either synchronously or asynchronously. They are defined as generic classes typed by their memory structure.

#### Key Features:
- Typed memory management
- Thread-safe operations
- Workflow integration
- Async processing

### Workflows (Aitorflows)
Workflows define processing pipelines as directed acyclic graphs of tasks.

#### Key Features:
- Task dependency management
- Parallel task execution
- Input/output validation
- Visualization tools

### Tasks
Tasks are the building blocks of workflows, encapsulating individual operations.

#### Key Features:
- Simple `>>` operator for defining dependencies
- Flexible input/output handling
- Decorator support for clean syntax

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.