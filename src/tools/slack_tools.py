"""
Slack messaging tools for ReAct agents.
Provides capabilities to send messages, files, and interact with Slack workspaces.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from aitor.tools import tool


@tool(
    name="slack_send_message",
    description="Send a message to a Slack channel or user. Supports text messages, mentions, and basic formatting.",
    timeout=30.0,
)
def slack_send_message(
    text: str,
    channel: str,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    thread_ts: Optional[str] = None,
    reply_broadcast: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Send a message to a Slack channel or user.

    Args:
        text: The message text to send (supports Slack markdown)
        channel: Channel name (e.g., '#general') or user ID (e.g., '@john') or channel ID
        username: Custom username for the bot (optional)
        icon_emoji: Custom emoji icon (e.g., ':robot_face:')
        thread_ts: Timestamp of parent message to reply to (for threading)
        reply_broadcast: Whether to broadcast threaded reply to channel

    Returns:
        Dictionary containing message response and metadata

    Raises:
        ValueError: If bot token not configured or invalid parameters
        requests.RequestException: If Slack API request fails
    """
    # Get bot token from environment
    bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")

    # Validate parameters
    if not text.strip():
        raise ValueError("Message text cannot be empty")

    if not channel.strip():
        raise ValueError("Channel cannot be empty")

    # Clean channel name (remove # if present)
    if channel.startswith("#"):
        channel = channel[1:]

    # Build API request
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "channel": channel,
        "text": text,
    }

    # Add optional parameters
    if username:
        payload["username"] = username
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji
    if thread_ts:
        payload["thread_ts"] = thread_ts
        if reply_broadcast:
            payload["reply_broadcast"] = True

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()

        data = response.json()

        if not data.get("ok", False):
            error_msg = data.get("error", "Unknown Slack API error")
            raise ValueError(f"Slack API error: {error_msg}")

        # Extract message info
        message = data.get("message", {})

        return {
            "success": True,
            "channel": data.get("channel"),
            "timestamp": message.get("ts"),
            "message_text": message.get("text", text),
            "user": message.get("user"),
            "username": message.get("username"),
            "thread_ts": message.get("thread_ts"),
            "permalink": f"https://slack.com/archives/{data.get('channel')}/p{message.get('ts', '').replace('.', '')}",
            "api_response": data,
            "sent_at": time.time(),
        }

    except requests.RequestException as e:
        raise requests.RequestException(f"Slack API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Slack API response: {str(e)}")


@tool(
    name="slack_send_rich_message",
    description="Send a rich message with blocks, attachments, or formatted content to Slack.",
    timeout=30.0,
)
def slack_send_rich_message(
    channel: str,
    text: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
    thread_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a rich message with blocks, attachments, or formatted content.

    Args:
        channel: Channel name (e.g., '#general') or user ID or channel ID
        text: Fallback text for notifications (required if no blocks)
        blocks: List of Block Kit blocks for rich formatting
        attachments: List of legacy attachments (deprecated but supported)
        username: Custom username for the bot
        icon_emoji: Custom emoji icon
        thread_ts: Timestamp of parent message to reply to

    Returns:
        Dictionary containing message response and metadata
    """
    # Get bot token from environment
    bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")

    # Validate parameters
    if not channel.strip():
        raise ValueError("Channel cannot be empty")

    if not text and not blocks and not attachments:
        raise ValueError("Must provide text, blocks, or attachments")

    # Clean channel name
    if channel.startswith("#"):
        channel = channel[1:]

    # Build API request
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "channel": channel,
    }

    # Add content
    if text:
        payload["text"] = text
    if blocks:
        payload["blocks"] = blocks
    if attachments:
        payload["attachments"] = attachments

    # Add optional parameters
    if username:
        payload["username"] = username
    if icon_emoji:
        payload["icon_emoji"] = icon_emoji
    if thread_ts:
        payload["thread_ts"] = thread_ts

    try:
        # Make API request
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()

        data = response.json()

        if not data.get("ok", False):
            error_msg = data.get("error", "Unknown Slack API error")
            raise ValueError(f"Slack API error: {error_msg}")

        # Extract message info
        message = data.get("message", {})

        return {
            "success": True,
            "channel": data.get("channel"),
            "timestamp": message.get("ts"),
            "message_text": message.get("text", text),
            "blocks": message.get("blocks"),
            "attachments": message.get("attachments"),
            "user": message.get("user"),
            "username": message.get("username"),
            "thread_ts": message.get("thread_ts"),
            "permalink": f"https://slack.com/archives/{data.get('channel')}/p{message.get('ts', '').replace('.', '')}",
            "api_response": data,
            "sent_at": time.time(),
        }

    except requests.RequestException as e:
        raise requests.RequestException(f"Slack API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Slack API response: {str(e)}")


@tool(
    name="slack_send_file",
    description="Upload and send a file to a Slack channel with optional message.",
    timeout=60.0,
)
def slack_send_file(
    file_path: str,
    channel: str,
    title: Optional[str] = None,
    initial_comment: Optional[str] = None,
    thread_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload and send a file to a Slack channel.

    Args:
        file_path: Path to the file to upload
        channel: Channel name (e.g., '#general') or user ID or channel ID
        title: Title for the file
        initial_comment: Message to accompany the file
        thread_ts: Timestamp of parent message to reply to

    Returns:
        Dictionary containing file upload response and metadata
    """
    # Get bot token from environment
    bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")

    # Validate parameters
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    if not channel.strip():
        raise ValueError("Channel cannot be empty")

    # Clean channel name
    if channel.startswith("#"):
        channel = channel[1:]

    # Build API request
    url = "https://slack.com/api/files.upload"
    headers = {
        "Authorization": f"Bearer {bot_token}",
    }

    # Prepare form data
    data = {
        "channels": channel,
    }

    if title:
        data["title"] = title
    if initial_comment:
        data["initial_comment"] = initial_comment
    if thread_ts:
        data["thread_ts"] = thread_ts

    try:
        # Open and upload file
        with open(file_path, "rb") as file:
            files = {
                "file": (os.path.basename(file_path), file, "application/octet-stream")
            }

            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=55
            )
            response.raise_for_status()

        response_data = response.json()

        if not response_data.get("ok", False):
            error_msg = response_data.get("error", "Unknown Slack API error")
            raise ValueError(f"Slack API error: {error_msg}")

        # Extract file info
        file_info = response_data.get("file", {})

        return {
            "success": True,
            "file_id": file_info.get("id"),
            "file_name": file_info.get("name"),
            "file_title": file_info.get("title"),
            "file_size": file_info.get("size"),
            "file_type": file_info.get("filetype"),
            "permalink": file_info.get("permalink"),
            "permalink_public": file_info.get("permalink_public"),
            "channel": channel,
            "timestamp": file_info.get("timestamp"),
            "thread_ts": thread_ts,
            "api_response": response_data,
            "uploaded_at": time.time(),
        }

    except requests.RequestException as e:
        raise requests.RequestException(f"Slack file upload failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Slack API response: {str(e)}")
    except IOError as e:
        raise IOError(f"Failed to read file {file_path}: {str(e)}")


@tool(
    name="slack_send_dm",
    description="Send a direct message to a Slack user by username or user ID.",
    timeout=30.0,
)
def slack_send_dm(
    text: str,
    user: str,
    username: Optional[str] = None,
    icon_emoji: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a direct message to a Slack user.

    Args:
        text: The message text to send
        user: Username (e.g., 'john.doe') or user ID (e.g., 'U1234567890')
        username: Custom username for the bot
        icon_emoji: Custom emoji icon

    Returns:
        Dictionary containing message response and metadata
    """
    # Get bot token from environment
    bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")

    # Validate parameters
    if not text.strip():
        raise ValueError("Message text cannot be empty")

    if not user.strip():
        raise ValueError("User cannot be empty")

    # Remove @ if present
    if user.startswith("@"):
        user = user[1:]

    try:
        # First, open a DM channel with the user
        dm_url = "https://slack.com/api/conversations.open"
        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json",
        }

        dm_payload = {
            "users": user,
        }

        dm_response = requests.post(
            dm_url, headers=headers, json=dm_payload, timeout=25
        )
        dm_response.raise_for_status()

        dm_data = dm_response.json()

        if not dm_data.get("ok", False):
            error_msg = dm_data.get("error", "Unknown Slack API error")
            raise ValueError(f"Failed to open DM channel: {error_msg}")

        # Get the DM channel ID
        dm_channel_id = dm_data["channel"]["id"]

        # Now send the message using the regular message sending
        return slack_send_message(
            text=text,
            channel=dm_channel_id,
            username=username,
            icon_emoji=icon_emoji,
        )

    except requests.RequestException as e:
        raise requests.RequestException(f"Slack DM failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Slack API response: {str(e)}")


@tool(
    name="slack_get_channels",
    description="Get list of channels in the Slack workspace.",
    timeout=30.0,
)
def slack_get_channels(
    types: Optional[str] = "public_channel,private_channel",
    limit: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Get list of channels in the Slack workspace.

    Args:
        types: Types of conversations to include (default: "public_channel,private_channel")
        limit: Maximum number of channels to return (default: 100)

    Returns:
        Dictionary containing list of channels and metadata
    """
    # Get bot token from environment
    bot_token = os.getenv("SLACK_BOT_TOKEN")

    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN environment variable not set")

    # Build API request
    url = "https://slack.com/api/conversations.list"
    headers = {
        "Authorization": f"Bearer {bot_token}",
        "Content-Type": "application/json",
    }

    params = {
        "types": types,
        "limit": min(limit or 100, 1000),
    }

    try:
        # Make API request
        response = requests.get(url, headers=headers, params=params, timeout=25)
        response.raise_for_status()

        data = response.json()

        if not data.get("ok", False):
            error_msg = data.get("error", "Unknown Slack API error")
            raise ValueError(f"Slack API error: {error_msg}")

        # Extract and format channel info
        channels = []
        for channel in data.get("channels", []):
            channels.append(
                {
                    "id": channel.get("id"),
                    "name": channel.get("name"),
                    "is_channel": channel.get("is_channel", False),
                    "is_group": channel.get("is_group", False),
                    "is_private": channel.get("is_private", False),
                    "is_archived": channel.get("is_archived", False),
                    "is_member": channel.get("is_member", False),
                    "num_members": channel.get("num_members", 0),
                    "topic": channel.get("topic", {}).get("value", ""),
                    "purpose": channel.get("purpose", {}).get("value", ""),
                    "created": channel.get("created"),
                }
            )

        return {
            "success": True,
            "channels": channels,
            "num_channels": len(channels),
            "response_metadata": data.get("response_metadata", {}),
            "retrieved_at": time.time(),
        }

    except requests.RequestException as e:
        raise requests.RequestException(f"Slack API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Slack API response: {str(e)}")
