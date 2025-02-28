import asyncio
from typing import List, Optional
from aitor.aitor import Aitor
from aitor.aitorflows import Aitorflow
from aitor.task import task


# Create a concrete implementation of Aitor
class TextProcessingAitor(Aitor[List[str]]):
    def __init__(self, name: Optional[str] = None):
        # Initialize with empty list as memory
        super().__init__(initial_memory=[], name=name)
    
    async def on_receive(self, message: str):
        # Store message in memory
        current_memory = self.get_memory()
        current_memory.append(message)
        self.set_memory(current_memory)
        
        # Process the message
        processed_msg = f"Processed: {message.upper()}"
        print(f"[{self.name}] {processed_msg}")
        
        # If workflow is attached, run it with the message
        if self.workflow:
            return await asyncio.to_thread(self.workflow.execute, message)
        
        return processed_msg


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
    # Create an aitor
    aitor = TextProcessingAitor(name="TextAitor")
    
    # Use the blocking 'ask' method
    result1 = await aitor.ask("Hello, world!")
    print(f"Ask result: {result1}")
    
    # Use the non-blocking 'tell' method
    aitor.tell("This is a non-blocking call")
    
    # Create and attach a workflow
    workflow = Aitorflow(name="Text Analysis")
    clean = clean_text
    word_counter = count_words
    sentiment = analyze_sentiment
    
    # Define workflow dependencies
    clean >> [word_counter, sentiment]
    
    # Add tasks to workflow
    workflow.add_task(clean)
    
    # Attach workflow to aitor
    aitor.attach_workflow(workflow)
    
    # Use 'ask' with the workflow
    result2 = await aitor.ask("This is a great example!")
    print(f"Workflow result: {result2}")
    
    # Check aitor's memory
    print(f"Aitor memory: {aitor.get_memory()}")
    
    # Always shutdown when done
    Aitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 