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
import asyncio
from typing import List
from aitor import Aitor

async def text_processor_handler(message: str, aitor: Aitor[List[str]]):
    # Store message in memory
    current_memory = aitor.get_memory()
    current_memory.append(message)
    aitor.set_memory(current_memory)
    
    # Process the message
    processed_msg = f"Processed: {message.upper()}"
    print(f"[{aitor.name}] {processed_msg}")
    
    # If workflow is attached, run it with the message
    if aitor.workflow:
        return await asyncio.to_thread(aitor.workflow.execute, message)
    
    return processed_msg

async def main():
    # Create an aitor directly
    aitor = Aitor(
        initial_memory=[],  # Start with empty list
        name="TextAitor",
        on_receive_handler=text_processor_handler
    )
    
    # Use the blocking 'ask' method
    result1 = await aitor.ask("Hello, world!")
    print(f"Ask result: {result1}")
    
    # Use the non-blocking 'tell' method
    aitor.tell("This is a non-blocking call")
    
    # Always shutdown when done
    Aitor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Building Workflows
```python
from aitor.aitorflows import Aitorflow
from aitor.task import task

# Define some tasks for a workflow
@task
def clean_text(text: str) -> str:
    return text.strip()

@task
def count_words(text: str) -> dict:
    return {"word_count": len(text.split())}

@task
def analyze_sentiment(text: str) -> dict:
    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "happy"]
    negative_words = ["bad", "terrible", "sad", "unhappy"]
    
    score = 0
    for word in text.lower().split():
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    
    return {"sentiment_score": score}

# Create and define workflow
workflow = Aitorflow(name="Text Analysis")
clean = clean_text
word_counter = count_words
sentiment = analyze_sentiment

# Define workflow dependencies
clean >> [word_counter, sentiment]

# Add tasks to workflow
workflow.add_task(clean)
workflow.add_task(word_counter)
workflow.add_task(sentiment)
```

### Attaching Workflows to Aitors
```python
async def main():
    aitor = Aitor(
        initial_memory=[],
        name="TextAitor",
        on_receive_handler=text_processor_handler
    )
    
    # Attach workflow
    aitor.attach_workflow(workflow)
    
    # Use 'ask' with the workflow
    result2 = await aitor.ask("This is a great example!")
    print(f"Workflow result: {result2}")
    
    # Check aitor's memory
    print(f"Aitor memory: {aitor.get_memory()}")
    
    Aitor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
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