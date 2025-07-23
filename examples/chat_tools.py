"""
Interactive chat tools for stdin/stdout communication.
"""

from aitor.tools import Tool


def get_user_input(prompt: str) -> str:
    """Get input from the user with a prompt."""
    if not prompt:
        raise ValueError("No prompt provided")

    # Print the prompt and get user input
    print(f"\n🤖 {prompt}")
    user_input = input("👤 ")

    return user_input


def send_message(message: str, message_type: str = "response") -> str:
    """Send a message to the user via stdout."""
    if not message:
        raise ValueError("No message provided")

    # Format message based on type
    emoji_map = {
        "response": "💬",
        "question": "❓",
        "information": "ℹ️",
        "call_to_action": "🎯",
    }

    emoji = emoji_map.get(message_type, "💬")

    # Send message to stdout
    print(f"\n{emoji} Agent: {message}")

    return f"Message sent: {message}"


def book_demo_call(
    user_name: str,
    user_email: str,
    preferred_time: str = "any time",
    interest_area: str = "general product demo",
) -> str:
    """Book a product demo call for the user."""
    if not user_name or not user_email:
        raise ValueError("User name and email are required to book a demo")

    # Simulate booking process
    booking_id = f"DEMO-{user_name[:3].upper()}-{hash(user_email) % 1000:03d}"

    confirmation_message = f"""
🎉 Demo call booked successfully!

📅 Booking Details:
   • Booking ID: {booking_id}
   • Name: {user_name}
   • Email: {user_email}
   • Preferred Time: {preferred_time}
   • Focus Area: {interest_area}

📧 You'll receive a calendar invite at {user_email} within the next few minutes.
📞 Our sales team will contact you to confirm the exact time.

Thank you for your interest in our product!
"""

    print(confirmation_message)

    return f"Demo call booked for {user_name} ({user_email})"


# Create tool instances
CHAT_TOOLS = [
    Tool(
        name="get_user_input",
        func=get_user_input,
        description="Get input from the user. Use this to ask questions or request information from the user.",
        parameters={
            "prompt": {
                "type": "str",
                "description": "The question or prompt to show to the user",
                "required": True,
            }
        },
    ),
    Tool(
        name="send_message",
        func=send_message,
        description="Send a message to the user. Use this to communicate responses, ask follow-up questions, or provide information.",
        parameters={
            "message": {
                "type": "str",
                "description": "The message to send to the user",
                "required": True,
            },
            "message_type": {
                "type": "str",
                "description": "Type of message: 'response', 'question', 'information', 'call_to_action'",
                "required": False,
            },
        },
    ),
    Tool(
        name="book_demo_call",
        func=book_demo_call,
        description="Book a product demo call for the user. Use this when the user expresses interest in scheduling a demo.",
        parameters={
            "user_name": {
                "type": "str",
                "description": "The user's name",
                "required": True,
            },
            "user_email": {
                "type": "str",
                "description": "The user's email address",
                "required": True,
            },
            "preferred_time": {
                "type": "str",
                "description": "User's preferred time for the demo",
                "required": False,
            },
            "interest_area": {
                "type": "str",
                "description": "Specific product area or feature the user is interested in",
                "required": False,
            },
        },
    ),
]
