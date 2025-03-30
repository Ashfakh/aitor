import asyncio
from aitor.aitor import Aitor
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


async def main():
    # Create an aitor directly
    aitor = Aitor(
        initial_memory=[],  # Start with empty list
        name="TextAitor",
    )
        
    # Define the workflow dependencies
    clean = clean_text
    word_counter = count_words
    sentiment = analyze_sentiment
    
    clean >> [word_counter, sentiment]
    
    # Create a workflow
    aitor.create_workflow(clean)

    aitor.workflow.print()
    
    # Use 'ask' with the workflow
    result2 = await aitor.ask("This is a great example!")
    print(f"Workflow result: {result2}")
    
    # Check aitor's memory
    print(f"Aitor memory: {aitor.get_memory()}")
    
    # Always shutdown when done
    Aitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())