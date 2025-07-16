"""
Sales Chat Agent - Interactive ReAct agent for sales conversations.
Uses stdin/stdout tools to have real-time conversations with users.
"""

import asyncio
import logging
from aitor.react_agent import ReactAgentBuilder
from aitor.llm import LLMManager, Message
from aitor.tools import ToolRegistry
from aitor.logging_config import setup_aitor_logging
from chat_tools import CHAT_TOOLS, send_message, book_demo_call, get_user_input

# Setup logging
setup_aitor_logging("INFO")
logger = logging.getLogger(__name__)


async def create_sales_agent():
    """Create a sales chat agent with interactive capabilities."""
    
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
    
    # Create agent with sales-focused system prompt
    tool_registry = ToolRegistry()
    agent = await (ReactAgentBuilder()
                   .name("SalesAgent")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(15)
                   .max_errors(3)
                   .system_prompt("""You are Sam, a friendly and knowledgeable sales representative for BrowserStack, the leading cross-browser testing platform.

PRODUCT OVERVIEW:
BrowserStack helps developers and QA teams test web and mobile applications across 3000+ real browsers and devices in the cloud. Key features:
- Test on real browsers and devices, not emulators
- Instant access to latest and legacy browser versions
- Automated testing with Selenium, Cypress, Playwright
- Live debugging and testing capabilities
- CI/CD integration for continuous testing
- Parallel testing to reduce execution time by 10x

YOUR ROLE:
- Answer questions about BrowserStack naturally and helpfully
- Understand the user's testing needs and pain points
- Subtly guide conversations toward scheduling a product demo
- Be conversational, not pushy - build genuine rapport
- Ask thoughtful follow-up questions to understand their testing setup

CONVERSATION FLOW:
1. Start by asking what brings them to BrowserStack today
2. Listen to their testing needs and ask clarifying questions
3. Share relevant product benefits that address their specific testing challenges
4. When appropriate, suggest a demo to show how BrowserStack can help their specific testing use case
5. If they show interest, help them book a demo call

TOOLS USAGE:
- Use get_user_input to ask questions and get responses
- Use send_message to share information and responses  
- Use book_demo_call when they want to schedule a demo

STYLE:
- Be warm, professional, and genuinely helpful
- Use the user's name when you learn it
- Ask open-ended questions to understand their testing challenges
- Share specific examples of how BrowserStack helps similar development teams
- Focus on value, not features

Remember: Your goal is to have a natural conversation that leads to a demo booking, not to overwhelm with information.""")
                   .build())
    
    # Register chat tools
    for tool in CHAT_TOOLS:
        await agent.register_tool(tool)
    
    return agent


async def run_sales_conversation():
    """Run a simple direct sales conversation without complex ReAct loops."""
    
    print("💬 BrowserStack Sales Chat")
    print("=" * 60)
    print("Chat with our AI sales agent! Type 'quit' to exit.")
    print("=" * 60)
    
    # Setup LLM
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 300
        }
    )
    
    # Start conversation
    send_message("Hello! I'm Sam from BrowserStack. What brings you here today?", "question")
    
    conversation_history = [
        Message("system", """You are Sam, a sales agent for BrowserStack - the leading cross-browser testing platform.

Your goal: Have a natural conversation that leads to booking a demo.

BrowserStack helps developers and QA teams:
- Test web and mobile apps across 3000+ real browsers and devices
- Run automated tests on real devices in the cloud
- Debug issues with instant access to live browsers
- Integrate seamlessly with CI/CD pipelines
- Reduce testing time by 10x with parallel testing
- Ensure pixel-perfect experiences across all platforms

Key pain points BrowserStack solves:
- Manual testing across multiple browsers is time-consuming
- Setting up device labs is expensive and complex
- Cross-browser bugs slip into production
- Testing mobile apps on real devices is challenging
- Slow feedback loops in development cycles

Conversation style:
- Be friendly and technical when appropriate
- Ask about their current testing setup and challenges
- Show how BrowserStack can solve their specific testing pain points
- When they show interest, suggest scheduling a demo

Guidelines:
- Ask about their tech stack and testing process
- Share relevant benefits based on their testing needs
- Be conversational, not pushy
- Guide toward booking a demo when appropriate""")
    ]
    
    try:
        while True:
            # Get user input
            user_input = get_user_input("Your response:")
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                send_message("Thank you for your time! Feel free to reach out anytime at sales@browserstack.com", "response")
                break
            
            # Add user message to history
            conversation_history.append(Message("user", user_input))
            
            # Get AI response
            llm = llm_manager.get_llm("openai")
            response = await llm.generate(conversation_history)
            
            # Add AI response to history
            conversation_history.append(Message("assistant", response.content))
            
            # Send AI response
            send_message(response.content, "response")
            
            # Check if user is interested in a demo
            if any(word in user_input.lower() for word in ['demo', 'schedule', 'book', 'show me', 'interested', 'call']):
                # Try to get contact info for demo booking
                name_input = get_user_input("Great! What's your name?")
                email_input = get_user_input("And your email address?")
                interest_input = get_user_input("What specific area interests you most? (e.g., cross-browser testing, mobile testing, automation)")
                
                # Book the demo
                book_demo_call(name_input, email_input, "this week", interest_input)
                
                send_message("Perfect! Our team will be in touch soon to schedule your personalized demo. Looking forward to showing you how BrowserStack can streamline your testing workflow!", "call_to_action")
                break
                
    except KeyboardInterrupt:
        send_message("Thanks for chatting! Have a great day!", "response")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Sales chat error: {e}", exc_info=True)


async def demo_conversation():
    """Run a simulated demo conversation for testing."""
    print("\n🎭 Demo Sales Conversation")
    print("=" * 60)
    
    # Simulate conversation flow without complex agent loops
    send_message("Hello! I'm Sam from BrowserStack. What brings you here today?", "question")
    
    print("👤 User: Hi! I'm Mike, a QA engineer. We're having issues with cross-browser testing taking forever.")
    
    send_message("I totally understand, Mike! Cross-browser testing can be a real pain point. How are you currently handling testing across different browsers and devices?", "question")
    
    print("👤 User: We have a few physical devices and use local browser instances, but it's slow and we keep missing edge cases.")
    
    send_message("That's exactly what BrowserStack solves! We provide instant access to 3000+ real browsers and devices in the cloud. You could run your entire test suite in parallel across multiple browsers simultaneously. Would you like to see how this works in a live demo?", "call_to_action")
    
    print("👤 User: Yes, that sounds exactly what we need! How do we set that up?")
    
    send_message("Perfect! Let me get your details to schedule a personalized demo.", "response")
    
    # Book demo
    book_demo_call("Mike Chen", "mike@techstartup.com", "this week", "cross-browser testing and automation")
    
    send_message("Excellent! Our team will reach out within 24 hours to schedule your demo. We'll show you exactly how other teams have reduced their testing time by 80% while catching more bugs!", "call_to_action")
    
    print("\n🎉 Demo conversation completed!")


async def simple_tool_demo():
    """Simple demonstration of sales agent tool usage."""
    print("\n🛠️ Simple Tool Usage Demo")
    print("=" * 60)
    
    # Just test the tools directly first
    print("Testing tools directly:")
    
    # Test send_message tool
    print("\n1. Testing send_message tool:")
    result1 = send_message("Hello! I'm Sam from BrowserStack. What brings you here today?", "question")
    print(f"Result: {result1}")
    
    # Test book_demo_call tool
    print("\n2. Testing book_demo_call tool:")
    result2 = book_demo_call("Sarah Johnson", "sarah@agency.com", "this week", "cross-browser testing")
    print(f"Result: {result2}")
    
    print("\n🎉 Direct tool testing completed!")
    
    # Now test with a very simple agent
    print("\n--- Testing with Simple Agent ---")
    
    # Setup LLM provider
    llm_manager = LLMManager(default_provider="openai")
    llm_manager.add_provider(
        name="openai",
        provider="openai",
        config={
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 200
        }
    )
    
    # Create very simple agent
    tool_registry = ToolRegistry()
    agent = await (ReactAgentBuilder()
                   .name("SimpleDemo")
                   .llm_manager(llm_manager)
                   .tool_registry(tool_registry)
                   .max_reasoning_steps(3)
                   .system_prompt("""You are Sam, a BrowserStack sales agent. When asked to send a message, immediately use:
ACT: send_message(message="your message here")

When asked to book a demo, immediately use:
ACT: book_demo_call(user_name="name", user_email="email")

Be direct. Use tools immediately. No thinking.""")
                   .build())
    
    # Register only send_message tool for simplicity
    await agent.register_tool(CHAT_TOOLS[1])  # send_message tool
    await agent.register_tool(CHAT_TOOLS[2])  # book_demo_call tool
    
    print(f"Agent tools: {agent.get_available_tools()}")
    
    try:
        print("\n📌 Test 1: Send greeting message")
        response1 = await agent.solve("Send a greeting message saying 'Hello from BrowserStack!'")
        print(f"Response: {response1}")
        
        print("\n📌 Test 2: Book a demo")
        response2 = await agent.solve("Book a demo for John Doe, email john@example.com")
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n🎉 Simple agent demo completed!")


async def main():
    """Main function to run sales chat agent."""
    print("🤖 BrowserStack Sales Agent")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Start interactive sales chat")
        print("2. Run demo conversation (simulated)")
        print("3. Run simple tool usage demo")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            await run_sales_conversation()
        elif choice == "2":
            await demo_conversation()
        elif choice == "3":
            await simple_tool_demo()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    asyncio.run(main())