"""
Planning ReAct Agent Demo - Mathematical tasks using calculator tool.
"""

import asyncio
import logging
from aitor.planning_agent import PlanningReactAgentBuilder
from aitor.llm import LLMManager
from aitor.tools import ToolRegistry
from aitor.todo import TodoStatus
from aitor.logging_config import setup_aitor_logging
from example_tools import EXAMPLE_TOOLS

# Setup enhanced logging
setup_aitor_logging("INFO")
logger = logging.getLogger(__name__)


async def math_planning_demo():
    """Demo showcasing mathematical problem planning."""
    print("=== Mathematical Planning Demo ===\n")
    
    # Setup LLM provider
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={
            "model": "o3-mini",
            "max_tokens": 1000
        }
    )
    
    # Create planning agent with mathematical focus
    tool_registry = ToolRegistry()
    agent = await (PlanningReactAgentBuilder()
                   .name("MathPlanningAgent")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(25)
                   .max_errors(3)
                   .build())
    
    # Register only calculator tool for focused demo
    await agent.register_tool(EXAMPLE_TOOLS[0])  # calculator
    
    print(f"Agent: {agent}")
    print(f"Tools: {agent.get_available_tools()}\n")
    
    # Test one focused mathematical problem
    problem = "Calculate 150 * 25 + 75 * 12 - 200, then divide the result by 10"
    
    print(f"PROBLEM: {problem}")
    print("="*80)
    
    try:
        # Execute planning
        response = await agent.solve(problem)
        
        print("\nFINAL ANSWER:")
        print("-"*40)
        print(response)
        
        # Show planning details
        print("\nPLANNING DETAILS:")
        print("-"*40)
        
        # Show todos
        todos = await agent.get_current_todos()
        print(f"Todos created: {len(todos)}")
        
        completed = await agent.get_todos_by_status(TodoStatus.COMPLETED)
        failed = await agent.get_todos_by_status(TodoStatus.FAILED)
        
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        
        print("\nCompleted calculation steps:")
        for todo in completed:
            print(f"  ✓ {todo.content}")
            if todo.result:
                result_text = todo.result[:80] + "..." if len(todo.result) > 80 else todo.result
                print(f"    → {result_text}")
        
        if failed:
            print("\nFailed calculations:")
            for todo in failed:
                print(f"  ✗ {todo.content}")
                print(f"    Error: {todo.error}")
        
        # Show sub-agents used
        sub_agents = agent.get_sub_agents()
        if sub_agents:
            print(f"\nSub-agents used: {sub_agents}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return agent

async def main():
    """Run mathematical planning demonstrations."""
    print("🧮 Mathematical Planning ReAct Agent Demo")
    print("=" * 80)
    
    try:
        # Run focused math planning demo
        # agent = await math_planning_demo()
        
        # Run simple demo
        await simple_math_demo()
        
        print(f"\n{'='*80}")
        print("🎉 Mathematical planning demos completed!")
        print("\nKey Features Demonstrated:")
        print("- Multi-step mathematical problem breakdown")
        print("- Sequential calculation planning")
        print("- Calculator tool integration with sub-agents")
        print("- Step-by-step verification and reasoning")
        print("- Planning-based task execution")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())