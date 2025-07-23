"""
ReAct Agent Demo - Complete examples showcasing the framework capabilities.
"""

import asyncio
from aitor.react_agent import ReactAgentBuilder
from aitor.llm import LLMManager
from aitor.tools import ToolRegistry
from example_tools import EXAMPLE_TOOLS

# Setup enhanced logging
# setup_aitor_logging("INFO")
# Optional: Disable verbose logging
# disable_aitor_logging()


async def debug_demo():
    """Demo that shows reasoning steps and prompts."""
    print("=== ReAct Agent Debug Demo ===\n")

    # Setup LLM provider
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={"model": "gpt-4o", "temperature": 0.1, "max_tokens": 500},
    )

    # Create agent with dynamic prompt construction
    tool_registry = ToolRegistry()
    agent = await (
        ReactAgentBuilder()
        .name("DebugAgent")
        .llm_manager(llm_manager)
        .tool_registry(tool_registry)
        .max_reasoning_steps(8)
        .agent_role("mathematical calculation agent")
        .task_goal("Solve mathematical problems step by step using available tools")
        .additional_instructions(
            "Show detailed reasoning and use tools for all calculations"
        )
        .build()
    )

    # Register tools
    await agent.register_tool(EXAMPLE_TOOLS[0])  # calculator
    await agent.register_tool(EXAMPLE_TOOLS[4])  # current_time

    print(f"Agent: {agent}")
    print(f"Tools: {agent.get_available_tools()}\n")

    # Test one problem and show detailed reasoning
    problem = "Calculate 12 * 8"
    print(f"Problem: {problem}")
    print("=" * 50)

    try:
        response = await agent.solve(problem)
        print(f"Final Response: {response}")

        # Show detailed reasoning trace
        print("\n" + "=" * 50)
        print("DETAILED REASONING TRACE:")
        print("=" * 50)

        reasoning_trace = agent.get_reasoning_trace()
        print(reasoning_trace)

        # Show tool executions
        print("\n" + "=" * 50)
        print("TOOL EXECUTIONS:")
        print("=" * 50)

        tool_history = agent.get_tool_execution_history()
        print(tool_history)

        # Show conversation history
        print("\n" + "=" * 50)
        print("CONVERSATION HISTORY:")
        print("=" * 50)

        conversation = agent.get_conversation_history()
        print(conversation)

        # Show full context
        print("\n" + "=" * 50)
        print("FULL CONTEXT:")
        print("=" * 50)

        full_context = agent.get_full_context()
        print(full_context)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    return agent


async def main():
    """Run debug demonstration."""
    print("🚀 ReAct Agent Framework - Debug Demo")
    print("=" * 60)

    try:
        agent = await debug_demo()

        print("\n🎉 Debug demo completed!")
        print("\nMemory Stats:")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
