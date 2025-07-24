# Basic Workflow Tutorial

This tutorial covers the fundamentals of creating and executing workflows with the Aitor framework.

## Overview

Aitor workflows are built using:
- **Tasks**: Functions wrapped with the `@task` decorator
- **Dependencies**: Defined using the `>>` operator
- **Workflows**: `Aitorflow` objects that execute DAGs
- **Agents**: `Aitor` instances that manage memory and execution

## Step 1: Simple Task Creation

Start by creating basic tasks that process data:

```python
import asyncio
from typing import List, Dict
from aitor import Aitor, Aitorflow, task

@task
def extract_text(raw_data: str) -> str:
    """Extract and clean text from raw input."""
    # Remove extra whitespace and normalize
    return raw_data.strip().replace('\n', ' ').replace('\t', ' ')

@task  
def count_words(text: str) -> int:
    """Count words in the text."""
    return len(text.split())

@task
def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis."""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

@task
def create_summary(text: str, word_count: int, sentiment: str) -> Dict[str, any]:
    """Create a comprehensive summary."""
    return {
        "original_text": text[:100] + "..." if len(text) > 100 else text,
        "word_count": word_count,
        "sentiment": sentiment,
        "length_category": "short" if word_count < 10 else "medium" if word_count < 50 else "long",
        "processed_at": datetime.now().isoformat()
    }
```

## Step 2: Define Task Dependencies

Use the `>>` operator to create a processing pipeline:

```python
# Define the workflow dependencies
extract_text >> count_words >> create_summary
extract_text >> analyze_sentiment >> create_summary

# This creates a DAG:
# extract_text → count_words ↘
#             → analyze_sentiment → create_summary
```

## Step 3: Create and Execute Workflow

```python
async def run_text_analysis():
    # Create workflow
    workflow = Aitorflow(name="TextAnalysis")
    
    # Add the entry point task (other connected tasks are auto-discovered)
    workflow.add_task(extract_text)
    
    # Create agent with typed memory
    agent = Aitor[List[Dict]](
        initial_memory=[],
        name="TextAnalyzer"
    )
    
    # Attach workflow to agent
    agent.attach_workflow(workflow)
    
    # Process sample text
    sample_text = """
        This is an amazing product! I love how it works so smoothly.
        The design is excellent and the performance is wonderful.
        I would definitely recommend this to others.
    """
    
    # Execute workflow
    result = await agent.ask(sample_text)
    
    print("Workflow Results:")
    print(f"Extract Text: {result['extract_text']}")
    print(f"Word Count: {result['count_words']}")
    print(f"Sentiment: {result['analyze_sentiment']}")
    print(f"Summary: {result['create_summary']}")
    
    # Update agent memory
    memory = agent.get_memory()
    memory.append(result['create_summary'])
    agent.set_memory(memory)
    
    print(f"\nMemory now contains {len(agent.get_memory())} analyses")
    
    # Cleanup
    Aitor.shutdown()

if __name__ == "__main__":
    asyncio.run(run_text_analysis())
```

## Step 4: Advanced Workflow Patterns

### Parallel Processing

When tasks don't depend on each other, they execute in parallel:

```python
@task
def extract_keywords(text: str) -> List[str]:
    """Extract important keywords."""
    # Simple keyword extraction
    words = text.lower().split()
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) > 3 and word not in common_words]
    return list(set(keywords))[:10]  # Top 10 unique keywords

@task
def calculate_readability(text: str) -> float:
    """Calculate readability score."""
    words = len(text.split())
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences == 0:
        sentences = 1
    avg_words_per_sentence = words / sentences
    return min(10.0, max(1.0, 10 - (avg_words_per_sentence - 15) / 5))

# These tasks run in parallel since they only depend on extract_text
extract_text >> extract_keywords
extract_text >> calculate_readability
```

### Complex Dependencies

```python
@task
def final_report(
    summary: Dict, 
    keywords: List[str], 
    readability: float
) -> Dict[str, any]:
    """Create comprehensive final report."""
    return {
        **summary,
        "keywords": keywords,
        "readability_score": readability,
        "analysis_complete": True
    }

# Create complex dependency graph
extract_text >> analyze_sentiment
extract_text >> count_words
extract_text >> extract_keywords
extract_text >> calculate_readability

analyze_sentiment >> create_summary
count_words >> create_summary

create_summary >> final_report
extract_keywords >> final_report
calculate_readability >> final_report
```

### Conditional Execution

```python
@task
def needs_translation(text: str) -> bool:
    """Check if text needs translation."""
    # Simple language detection
    non_english_chars = sum(1 for char in text if ord(char) > 127)
    return non_english_chars > len(text) * 0.1

@task
def translate_text(text: str) -> str:
    """Translate text to English (mock implementation)."""
    # In real implementation, use translation service
    return f"[TRANSLATED] {text}"

# Dynamic workflow based on conditions
async def conditional_workflow():
    workflow = Aitorflow(name="ConditionalProcessing")
    
    # Custom handler that handles conditional logic
    async def smart_handler(message: str, agent: Aitor[List[Dict]]):
        # First check if translation is needed
        if needs_translation.func(message):
            translated = translate_text.func(message)
            # Process translated text
            result = await asyncio.to_thread(workflow.execute, translated)
        else:
            # Process original text
            result = await asyncio.to_thread(workflow.execute, message)
        
        # Update memory
        memory = agent.get_memory()
        memory.append(result)
        agent.set_memory(memory)
        
        return result
    
    # Set up workflow for main processing (without translation tasks)
    extract_text >> analyze_sentiment >> create_summary
    workflow.add_task(extract_text)
    
    # Create agent with custom handler
    agent = Aitor[List[Dict]](
        initial_memory=[],
        name="SmartProcessor",
        on_receive_handler=smart_handler
    )
    agent.attach_workflow(workflow)
    
    return agent
```

## Step 5: Error Handling and Validation

```python
@task
def validate_input(text: str) -> str:
    """Validate and sanitize input."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    if len(text) > 10000:
        raise ValueError("Input text too long (max 10,000 characters)")
    
    # Remove potentially harmful content
    sanitized = text.replace('<script>', '').replace('</script>', '')
    
    return sanitized

@task
def safe_extract_text(raw_data: str) -> str:
    """Extract text with error handling."""
    try:
        validated = validate_input.func(raw_data)
        return validated.strip().replace('\n', ' ').replace('\t', ' ')
    except ValueError as e:
        # Return error indicator that downstream tasks can handle
        return f"ERROR: {str(e)}"

@task
def safe_count_words(text: str) -> int:
    """Count words with error handling."""
    if text.startswith("ERROR:"):
        return 0
    return len(text.split())

# Use safe tasks in workflow
safe_extract_text >> safe_count_words
```

## Step 6: Workflow Visualization

```python
def visualize_workflow():
    """Generate Mermaid diagram of the workflow."""
    workflow = Aitorflow(name="CompleteTextAnalysis")
    
    # Set up complete workflow
    extract_text >> count_words >> create_summary
    extract_text >> analyze_sentiment >> create_summary
    extract_text >> extract_keywords >> final_report
    extract_text >> calculate_readability >> final_report
    create_summary >> final_report
    
    workflow.add_task(extract_text)
    
    # Generate visualization
    mermaid_diagram = workflow.visualize()
    print("Workflow Diagram (Mermaid):")
    print(mermaid_diagram)
    
    # Save to file
    with open("workflow_diagram.md", "w") as f:
        f.write("# Workflow Diagram\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_diagram)
        f.write("\n```")
    
    print("Diagram saved to workflow_diagram.md")

# Generate diagram
visualize_workflow()
```

## Complete Example

Here's a complete, runnable example that demonstrates all concepts:

```python
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from aitor import Aitor, Aitorflow, task

# Task definitions
@task
def validate_and_extract(raw_data: str) -> str:
    """Validate input and extract clean text."""
    if not raw_data or not raw_data.strip():
        raise ValueError("Input cannot be empty")
    return raw_data.strip().replace('\n', ' ').replace('\t', ' ')

@task
def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

@task
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

@task
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text."""
    words = text.lower().split()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'it'}
    keywords = [word.strip('.,!?;:"()[]') for word in words if len(word) > 3 and word not in stop_words]
    return list(set(keywords))[:10]

@task
def calculate_readability(text: str) -> float:
    """Calculate simple readability score."""
    words = len(text.split())
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    avg_words_per_sentence = words / sentences
    
    # Simple scoring: ideal is 15-20 words per sentence
    if 15 <= avg_words_per_sentence <= 20:
        return 10.0
    elif avg_words_per_sentence < 15:
        return max(1.0, 10 - (15 - avg_words_per_sentence) * 0.5)
    else:
        return max(1.0, 10 - (avg_words_per_sentence - 20) * 0.3)

@task
def create_summary(text: str, word_count: int, sentiment: str) -> Dict[str, Any]:
    """Create basic summary."""
    return {
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "word_count": word_count,
        "sentiment": sentiment,
        "length_category": "short" if word_count < 20 else "medium" if word_count < 100 else "long",
        "processed_at": datetime.now().isoformat()
    }

@task
def final_report(
    summary: Dict[str, Any],
    keywords: List[str],
    readability: float
) -> Dict[str, Any]:
    """Create comprehensive final report."""
    return {
        **summary,
        "keywords": keywords[:5],  # Top 5 keywords
        "readability_score": round(readability, 2),
        "readability_level": (
            "Easy" if readability >= 8 else
            "Medium" if readability >= 6 else
            "Difficult"
        ),
        "analysis_complete": True,
        "report_generated_at": datetime.now().isoformat()
    }

async def main():
    """Main execution function."""
    print("🚀 Starting Text Analysis Workflow\n")
    
    # Define workflow dependencies
    validate_and_extract >> count_words >> create_summary
    validate_and_extract >> analyze_sentiment >> create_summary
    validate_and_extract >> extract_keywords >> final_report
    validate_and_extract >> calculate_readability >> final_report
    create_summary >> final_report
    
    # Create workflow
    workflow = Aitorflow(name="TextAnalysisWorkflow")
    workflow.add_task(validate_and_extract)
    
    # Print workflow visualization
    print("📊 Workflow Structure:")
    print(workflow.visualize())
    print()
    
    # Create agent
    agent = Aitor[List[Dict[str, Any]]](
        initial_memory=[],
        name="TextAnalysisAgent"
    )
    agent.attach_workflow(workflow)
    
    # Test data
    test_texts = [
        """
        This product is absolutely amazing! The quality is excellent and the 
        customer service is wonderful. I'm very happy with my purchase and 
        would definitely recommend it to others. The delivery was fast and 
        the packaging was great too.
        """,
        """
        The experience was terrible. The product broke immediately and the 
        customer service was awful. I hate dealing with this company.
        """,
        """
        The weather today is nice. It's sunny and warm outside. Perfect 
        for a walk in the park.
        """
    ]
    
    # Process each text
    for i, text in enumerate(test_texts, 1):
        print(f"📝 Processing Text {i}:")
        print(f"Input: {text.strip()[:100]}...")
        print()
        
        try:
            # Execute workflow
            result = await agent.ask(text)
            
            # Display results
            final_result = result.get('final_report', {})
            print(f"✅ Analysis Complete:")
            print(f"   Word Count: {final_result.get('word_count', 'N/A')}")
            print(f"   Sentiment: {final_result.get('sentiment', 'N/A')}")
            print(f"   Readability: {final_result.get('readability_level', 'N/A')} ({final_result.get('readability_score', 'N/A')}/10)")
            print(f"   Keywords: {', '.join(final_result.get('keywords', []))}")
            print(f"   Category: {final_result.get('length_category', 'N/A')}")
            
            # Update agent memory
            memory = agent.get_memory()
            memory.append(final_result)
            agent.set_memory(memory)
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")
        
        print("-" * 60)
    
    # Display memory statistics
    memory = agent.get_memory()
    print(f"\n📊 Agent Memory: {len(memory)} analyses stored")
    
    if memory:
        sentiments = [analysis.get('sentiment', 'unknown') for analysis in memory]
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        print(f"   Sentiment Distribution: {sentiment_counts}")
    
    # Cleanup
    Aitor.shutdown()
    print("\n✨ Workflow complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Concepts Learned

1. **Task Creation**: Use `@task` decorator to wrap functions
2. **Dependencies**: Use `>>` operator to define execution order
3. **Parallel Execution**: Independent tasks run concurrently
4. **Memory Management**: Agents maintain typed memory across executions
5. **Error Handling**: Tasks can handle errors gracefully
6. **Visualization**: Generate Mermaid diagrams of workflows
7. **Workflow Execution**: Use `Aitorflow` for DAG-based processing

## Next Steps

- Explore [ReAct Agents](react-agent.md) for intelligent reasoning
- Learn about [Custom Tool Development](custom-tools.md) for external integrations
- Check out [Memory Management](memory-management.md) for advanced memory patterns

This tutorial provides a solid foundation for building workflows with the Aitor framework. The patterns shown here scale to much more complex use cases while maintaining clarity and maintainability.