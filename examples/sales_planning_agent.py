"""
Sales Planning Agent - Uses planning-based approach for structured sales conversations.
Demonstrates how planning agents break down sales tasks into organized todos.
"""

import asyncio
import logging
from aitor.planning_agent import PlanningReactAgentBuilder
from aitor.llm import LLMManager
from aitor.tools import ToolRegistry
from aitor.todo import TodoStatus
from aitor.logging_config import setup_aitor_logging
from chat_tools import CHAT_TOOLS

# Setup logging
setup_aitor_logging("INFO")
logger = logging.getLogger(__name__)


async def create_sales_planning_agent():
    """Create a sales planning agent that structures conversations into tasks."""
    
    # Setup LLM provider
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai", 
        config={
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 800
        }
    )
    
    # Create planning agent with sales-focused system prompt
    tool_registry = ToolRegistry()
    agent = await (PlanningReactAgentBuilder()
                   .name("SalesPlanningAgent")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(20)
                   .max_errors(3)
                   .system_prompt("""You are Sam, a strategic sales planning agent for BrowserStack, the leading cross-browser testing platform.

Your approach is to break down sales conversations into structured, sequential tasks that lead to successful demo bookings.

PRODUCT KNOWLEDGE:
BrowserStack provides:
- Testing on 3000+ real browsers and devices
- Automated testing with Selenium, Cypress, Playwright
- Live debugging and testing capabilities
- CI/CD integration for continuous testing
- Parallel testing to reduce execution time by 10x
- Mobile app testing on real devices

PLANNING APPROACH:
When given a sales objective, create a structured plan with these types of tasks:
1. DISCOVERY tasks - Learn about the prospect's current testing setup and pain points
2. QUALIFICATION tasks - Understand their budget, timeline, and decision-making process
3. PRESENTATION tasks - Share relevant BrowserStack benefits that address their specific needs
4. OBJECTION HANDLING tasks - Address any concerns or hesitations
5. CLOSING tasks - Guide toward booking a demo or next steps

TASK EXECUTION:
- Use get_user_input to gather information and ask discovery questions
- Use send_message to share product information and build rapport
- Use book_demo_call when ready to schedule a demo
- Each task should have a clear objective and expected outcome
- Be conversational and natural, not robotic

STYLE:
- Professional yet friendly tone
- Ask thoughtful, open-ended questions
- Listen actively to responses
- Tailor your approach based on their specific testing challenges
- Focus on value and outcomes, not just features""")
                   .build())
    
    # Register chat tools
    for tool in CHAT_TOOLS:
        await agent.register_tool(tool)
    
    return agent


async def run_structured_sales_conversation():
    """Run a structured sales conversation using planning approach."""
    
    print("🎯 BrowserStack Sales Planning Agent")
    print("=" * 60)
    print("Strategic sales conversation with structured task planning")
    print("=" * 60)
    
    # Create planning agent
    agent = await create_sales_planning_agent()
    
    # Define sales objective
    sales_objective = """
    Conduct a comprehensive sales conversation with a prospect who is interested in BrowserStack.
    
    Your goal is to:
    1. Understand their current testing challenges and setup
    2. Identify their specific pain points and needs
    3. Present relevant BrowserStack solutions
    4. Address any concerns or objections
    5. Guide them toward booking a product demo
    
    Be strategic and methodical in your approach, ensuring each interaction moves the conversation forward.
    """
    
    print("📋 Sales Objective:")
    print(sales_objective)
    print("\n" + "=" * 60)
    
    try:
        # Start the structured sales conversation
        print("🚀 Starting structured sales conversation...")
        response = await agent.solve(sales_objective)
        
        print("\n✅ CONVERSATION RESULT:")
        print("-" * 40)
        print(response)
        
        # Show planning breakdown
        print("\n📊 PLANNING BREAKDOWN:")
        print("-" * 40)
        
        # Show all todos created
        todos = await agent.get_current_todos()
        print(f"Total sales tasks planned: {len(todos)}")
        
        # Show completed tasks
        completed = await agent.get_todos_by_status(TodoStatus.COMPLETED)
        print(f"Tasks completed: {len(completed)}")
        
        if completed:
            print("\n✅ Completed sales tasks:")
            for todo in completed:
                print(f"  ✓ {todo.content}")
                if todo.result:
                    result_preview = todo.result[:100] + "..." if len(todo.result) > 100 else todo.result
                    print(f"    → {result_preview}")
        
        # Show failed tasks
        failed = await agent.get_todos_by_status(TodoStatus.FAILED)
        if failed:
            print(f"\n❌ Failed tasks: {len(failed)}")
            for todo in failed:
                print(f"  ✗ {todo.content}")
                if todo.error:
                    print(f"    Error: {todo.error}")
        
        # Show pending tasks
        pending = await agent.get_todos_by_status(TodoStatus.PENDING)
        if pending:
            print(f"\n⏳ Pending tasks: {len(pending)}")
            for todo in pending:
                print(f"  ○ {todo.content}")
        
        # Show sub-agents used
        sub_agents = agent.get_sub_agents()
        if sub_agents:
            print(f"\n🤖 Sub-agents used: {len(sub_agents)}")
            for sub_agent in sub_agents:
                print(f"  - {sub_agent}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def demo_planning_conversation():
    """Run a simulated planning conversation for demonstration."""
    
    print("\n🎭 Demo Planning Sales Conversation")
    print("=" * 60)
    
    # Create planning agent
    agent = await create_sales_planning_agent()
    
    # Simulate a structured sales approach
    objective = """
    Conduct a sales conversation with Mike, a QA engineer at a tech startup.
    He's mentioned they're struggling with cross-browser testing taking too long.
    
    Plan and execute a structured approach to:
    1. Understand his current testing setup
    2. Identify specific pain points
    3. Present BrowserStack solutions
    4. Guide toward booking a demo
    """
    
    print("📋 Demo Objective:")
    print(objective)
    print("\n" + "-" * 60)
    
    try:
        # Execute the planned conversation
        result = await agent.solve(objective)
        
        print("\n🎯 DEMO RESULT:")
        print("-" * 40)
        print(result)
        
        # Show the planning structure
        todos = await agent.get_current_todos()
        print("\n📈 SALES PLANNING STRUCTURE:")
        print("-" * 40)
        print(f"Total planned tasks: {len(todos)}")
        
        # Group todos by type/phase
        discovery_tasks = [t for t in todos if any(word in t.content.lower() for word in ['discover', 'understand', 'learn', 'ask', 'current'])]
        presentation_tasks = [t for t in todos if any(word in t.content.lower() for word in ['present', 'show', 'explain', 'benefit', 'solution'])]
        closing_tasks = [t for t in todos if any(word in t.content.lower() for word in ['demo', 'book', 'schedule', 'next', 'close'])]
        
        if discovery_tasks:
            print(f"\n🔍 Discovery Phase ({len(discovery_tasks)} tasks):")
            for task in discovery_tasks:
                status = "✓" if task.status == TodoStatus.COMPLETED else "✗" if task.status == TodoStatus.FAILED else "○"
                print(f"  {status} {task.content}")
        
        if presentation_tasks:
            print(f"\n📢 Presentation Phase ({len(presentation_tasks)} tasks):")
            for task in presentation_tasks:
                status = "✓" if task.status == TodoStatus.COMPLETED else "✗" if task.status == TodoStatus.FAILED else "○"
                print(f"  {status} {task.content}")
        
        if closing_tasks:
            print(f"\n🎯 Closing Phase ({len(closing_tasks)} tasks):")
            for task in closing_tasks:
                status = "✓" if task.status == TodoStatus.COMPLETED else "✗" if task.status == TodoStatus.FAILED else "○"
                print(f"  {status} {task.content}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()


def create_sales_strategy(target_audience: str, pain_points: str) -> str:
    """Tool to create a sales strategy based on target audience and pain points."""
    strategy = f"""
Sales Strategy for {target_audience}:

Pain Points Addressed: {pain_points}

Recommended Approach:
1. Discovery Phase: Understand current testing setup and challenges
2. Qualification Phase: Assess budget, timeline, and decision-making process  
3. Presentation Phase: Show how BrowserStack addresses their specific pain points
4. Demo Phase: Provide hands-on demonstration of relevant features
5. Closing Phase: Address objections and guide toward next steps

Key Messages:
- BrowserStack provides instant access to 3000+ real browsers and devices
- Reduces testing time by up to 10x with parallel testing
- Eliminates need for maintaining device labs
- Integrates seamlessly with existing CI/CD pipelines
"""
    return strategy


def identify_key_questions(role: str, industry: str = "technology") -> str:
    """Tool to identify key discovery questions for a specific role."""
    questions = f"""
Key Discovery Questions for {role} in {industry}:

Current State Questions:
- What testing tools and frameworks are you currently using?
- How much time does your team spend on cross-browser testing?
- What devices/browsers do you test on regularly?
- How do you handle mobile app testing?

Pain Point Questions:
- What are your biggest testing challenges right now?
- How do testing bottlenecks impact your release cycles?
- What issues have you encountered with current testing setup?
- How do you ensure consistent testing across different environments?

Goal Questions:
- What would ideal testing workflow look like for your team?
- What outcomes are you hoping to achieve?
- How do you measure testing efficiency?
- What would success look like 6 months from now?
"""
    return questions


def create_value_proposition(role: str, company_size: str = "medium") -> str:
    """Tool to create role-specific value proposition."""
    if "developer" in role.lower():
        value_prop = f"""
Value Proposition for {role} at {company_size} company:

For Developers:
- Instant access to real browsers without setup overhead
- Faster feedback loops with parallel testing
- Debug issues directly in live browsers
- Seamless integration with existing development tools
- Reduced time spent on environment setup and maintenance

ROI Benefits:
- 70% reduction in testing setup time
- 10x faster test execution with parallel runs
- 50% fewer production bugs through comprehensive testing
- Faster time-to-market for new features
"""
    else:
        value_prop = f"""
Value Proposition for {role} at {company_size} company:

For QA Teams:
- Comprehensive testing across all browsers and devices
- Automated testing capabilities with popular frameworks
- Real device testing for mobile applications
- Detailed test reports and analytics
- Integration with bug tracking and CI/CD tools

ROI Benefits:
- 80% faster testing cycles
- 90% reduction in device lab maintenance costs
- Improved test coverage and quality
- Better collaboration between dev and QA teams
"""
    
    return value_prop


async def quick_planning_demo():
    """Quick demonstration of planning agent capabilities."""
    
    print("\n⚡ Quick Planning Demo")
    print("=" * 60)
    
    # Setup simpler LLM config
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 500
        }
    )
    
    # Create focused planning agent
    tool_registry = ToolRegistry()
    agent = await (PlanningReactAgentBuilder()
                   .name("QuickSalesPlanner")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(15)
                   .system_prompt("""You are Sam from BrowserStack. 
                   
Plan a structured sales conversation that will lead to booking a demo.
Break it down into clear, actionable steps.

You have access to these tools:
- get_user_input(prompt): Ask questions to gather information from prospects
- send_message(message, message_type): Send messages and information to prospects
- book_demo_call(user_name, user_email, preferred_time, interest_area): Book demo calls
- create_sales_strategy(target_audience, pain_points): Create structured sales strategies
- identify_key_questions(role, industry): Generate key discovery questions
- create_value_proposition(role, company_size): Create tailored value propositions

Use these tools to create a comprehensive sales plan that involves discovery, presentation, and closing.
Start by using planning tools to create strategy, then use conversation tools to execute the plan.""")
                   .build())
    
    # Use the same chat tools that work in sales_chat_agent
    # Register all CHAT_TOOLS for proper sales conversation
    for tool in CHAT_TOOLS:
        await agent.register_tool(tool)
    
    # Also add the planning tools we defined above
    from aitor.tools import Tool
    planning_tools = [
        Tool(
            name="create_sales_strategy",
            func=create_sales_strategy,
            description="Create a structured sales strategy based on target audience and their pain points",
            parameters={
                "target_audience": {"type": "str", "description": "Target audience", "required": True},
                "pain_points": {"type": "str", "description": "Main pain points", "required": True}
            }
        ),
        Tool(
            name="identify_key_questions", 
            func=identify_key_questions,
            description="Generate key discovery questions for a specific role",
            parameters={
                "role": {"type": "str", "description": "Role of the prospect", "required": True},
                "industry": {"type": "str", "description": "Industry", "required": False}
            }
        ),
        Tool(
            name="create_value_proposition",
            func=create_value_proposition,
            description="Create a tailored value proposition for a specific role",
            parameters={
                "role": {"type": "str", "description": "Role of the prospect", "required": True},
                "company_size": {"type": "str", "description": "Company size", "required": False}
            }
        )
    ]
    
    for tool in planning_tools:
        await agent.register_tool(tool)
    
    # Simple sales planning task
    task = "Plan a comprehensive sales approach for a QA Engineer at a startup who is struggling with manual testing processes and wants to automate their workflows"
    
    print(f"📋 Task: {task}")
    print("-" * 40)
    
    try:
        result = await agent.solve(task)
        print(f"Result: {result}")
        
        # Show planning structure
        todos = await agent.get_current_todos()
        print(f"\n📊 Planning Structure ({len(todos)} tasks):")
        
        completed = await agent.get_todos_by_status(TodoStatus.COMPLETED)
        if completed:
            print(f"\n✅ Completed tasks ({len(completed)}):")
            for i, todo in enumerate(completed, 1):
                print(f"{i}. {todo.content}")
                if todo.result:
                    result_preview = todo.result[:100] + "..." if len(todo.result) > 100 else todo.result
                    print(f"   → {result_preview}")
        
        failed = await agent.get_todos_by_status(TodoStatus.FAILED)
        if failed:
            print(f"\n❌ Failed tasks ({len(failed)}):")
            for todo in failed:
                print(f"  - {todo.content}")
                if todo.error:
                    print(f"    Error: {todo.error}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to demonstrate sales planning agent capabilities."""
    
    print("🎯 BrowserStack Sales Planning Agent Demo")
    print("=" * 80)
    
    try:
        while True:
            print("\nChoose a demo:")
            print("1. Run structured sales conversation (interactive)")
            print("2. Run demo planning conversation (simulated)")
            print("3. Quick planning demonstration")
            print("4. Exit")
            
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n🚀 Running quick planning demo by default...")
                choice = "3"
            
            if choice == "1":
                await run_structured_sales_conversation()
            elif choice == "2":
                await demo_planning_conversation()
            elif choice == "3":
                await quick_planning_demo()
                break  # Exit after demo in non-interactive mode
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            
            print("\n" + "=" * 80)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())