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
    
    # Create planning agent
    tool_registry = ToolRegistry()
    agent = await (PlanningReactAgentBuilder()
                   .name("MathPlanningAgent")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(25)
                   .max_errors(3)
                   .system_prompt("""You are a mathematical planning agent that breaks down complex calculations into step-by-step tasks.

Your approach:
1. PLAN: Analyze the mathematical problem and create ordered calculation steps
2. EXECUTE: Work through calculations systematically using the calculator tool
3. VERIFY: Check intermediate results make sense
4. COMPLETE: Provide the final answer with clear reasoning

Key principles:
- Break complex math problems into simple calculations
- Use the calculator tool for all arithmetic operations
- Show your work step by step
- Verify results are reasonable""")
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
        
        print(f"\nFINAL ANSWER:")
        print("-"*40)
        print(response)
        
        # Show planning details
        print(f"\nPLANNING DETAILS:")
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


async def simple_math_demo():
    """Simple mathematical planning demo."""
    print(f"\n{'='*80}")
    print("SIMPLE MATH PLANNING DEMO")
    print("="*80)
    
    # Setup LLM provider
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={
            "model": "o3-mini",
            "max_tokens": 500
        }
    )
    
    # Create simpler planning agent
    tool_registry = ToolRegistry()
    agent = await (PlanningReactAgentBuilder()
                   .name("SimpleMathAgent")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(12)
                   .build())
    
    # Register calculator tool
    await agent.register_tool(EXAMPLE_TOOLS[0])  # calculator
    
    # Simple multi-step problem
    problem = "Calculate the total cost if I buy 3 items at $24.99 each, 2 items at $15.50 each, apply a 15% discount, then add 8.5% sales tax."
    
    print(f"Problem: {problem}")
    print("-" * 40)
    
    try:
        response = await agent.solve(problem)
        print(f"Final Answer: {response}")
        
        # Show planning breakdown
        todos = await agent.get_current_todos()
        print(f"\nCalculation steps planned: {len(todos)}")
        
        for todo in todos:
            status_symbol = "✓" if todo.status == TodoStatus.COMPLETED else "✗" if todo.status == TodoStatus.FAILED else "○"
            print(f"  {status_symbol} {todo.content}")
            if todo.result and todo.status == TodoStatus.COMPLETED:
                print(f"    → {todo.result}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run mathematical planning demonstrations."""
    print("🧮 Mathematical Planning ReAct Agent Demo")
    print("=" * 80)
    
    try:
        # Run focused math planning demo
        agent = await math_planning_demo()
        
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