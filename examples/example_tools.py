"""
Example tools for ReAct agents.
Demonstrates various tool types and usage patterns.
"""

import asyncio
import json
import math
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from aitor.tools import tool


# Mathematical tools
@tool(
    name="calculator",
    description="Perform basic mathematical calculations",
    timeout=5.0
)
def calculator(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            k: v for k, v in math.__dict__.items() 
            if not k.startswith("__")
        }
        allowed_names.update({
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow
        })
        
        # Remove dangerous functions
        forbidden = ["__", "import", "exec", "eval", "open", "file"]
        for term in forbidden:
            if term in expression:
                raise ValueError(f"Forbidden term '{term}' in expression")
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
        
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")


@tool(
    name="random_number",
    description="Generate a random number within a specified range",
    timeout=2.0
)
def random_number(min_val: int = 1, max_val: int = 100) -> int:
    """
    Generate a random integer between min_val and max_val (inclusive).
    
    Args:
        min_val: Minimum value (default: 1)
        max_val: Maximum value (default: 100)
        
    Returns:
        Random integer
    """
    if min_val > max_val:
        raise ValueError("min_val must be less than or equal to max_val")
    
    return random.randint(min_val, max_val)


# Text processing tools
@tool(
    name="text_analyzer",
    description="Analyze text and return various statistics",
    timeout=10.0
)
def text_analyzer(text: str) -> Dict[str, Any]:
    """
    Analyze text and return statistics.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with text analysis results
    """
    words = text.split()
    sentences = text.split('.')
    
    # Count characters
    char_count = len(text)
    char_count_no_spaces = len(text.replace(' ', ''))
    
    # Count words
    word_count = len(words)
    
    # Count sentences
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Most common words
    word_freq = {}
    for word in words:
        word_lower = word.lower().strip('.,!?;:"()[]{}')
        word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
    
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_word_length": round(avg_word_length, 2),
        "most_common_words": most_common
    }


@tool(
    name="text_transformer",
    description="Transform text in various ways (uppercase, lowercase, reverse, etc.)",
    timeout=5.0
)
def text_transformer(text: str, operation: str) -> str:
    """
    Transform text using various operations.
    
    Args:
        text: Text to transform
        operation: Type of transformation (uppercase, lowercase, reverse, title, capitalize)
        
    Returns:
        Transformed text
    """
    operation = operation.lower()
    
    if operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    elif operation == "title":
        return text.title()
    elif operation == "capitalize":
        return text.capitalize()
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Time and date tools
@tool(
    name="current_time",
    description="Get current date and time",
    timeout=2.0
)
def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current date and time.
    
    Args:
        format: Date format string (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        Formatted current date and time
    """
    return datetime.now().strftime(format)


@tool(
    name="timestamp_converter",
    description="Convert between timestamp and human-readable date",
    timeout=3.0
)
def timestamp_converter(value: str, from_format: str = "timestamp") -> str:
    """
    Convert between timestamp and human-readable date.
    
    Args:
        value: Value to convert (timestamp or date string)
        from_format: Format of input value ("timestamp" or "date")
        
    Returns:
        Converted value
    """
    if from_format == "timestamp":
        try:
            timestamp = float(value)
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Invalid timestamp format")
    elif from_format == "date":
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return str(int(dt.timestamp()))
        except ValueError:
            raise ValueError("Invalid date format")
    else:
        raise ValueError("from_format must be 'timestamp' or 'date'")


# File system tools
@tool(
    name="file_reader",
    description="Read content from a file",
    timeout=10.0
)
def file_reader(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")


@tool(
    name="file_writer",
    description="Write content to a file",
    timeout=10.0
)
def file_writer(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (default: utf-8)
        
    Returns:
        Success message
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        raise ValueError(f"Error writing file: {str(e)}")


@tool(
    name="directory_lister",
    description="List files and directories in a given path",
    timeout=5.0
)
def directory_lister(path: str = ".", show_hidden: bool = False) -> List[str]:
    """
    List files and directories in a path.
    
    Args:
        path: Directory path (default: current directory)
        show_hidden: Whether to show hidden files (default: False)
        
    Returns:
        List of file and directory names
    """
    try:
        items = os.listdir(path)
        if not show_hidden:
            items = [item for item in items if not item.startswith('.')]
        return sorted(items)
    except Exception as e:
        raise ValueError(f"Error listing directory: {str(e)}")


# Data processing tools
@tool(
    name="json_parser",
    description="Parse JSON string and extract specific fields",
    timeout=5.0
)
def json_parser(json_string: str, field_path: Optional[str] = None) -> Any:
    """
    Parse JSON string and optionally extract specific field.
    
    Args:
        json_string: JSON string to parse
        field_path: Dot-separated path to extract specific field (optional)
        
    Returns:
        Parsed JSON data or specific field value
    """
    try:
        data = json.loads(json_string)
        
        if field_path:
            # Navigate to specific field
            parts = field_path.split('.')
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                else:
                    raise ValueError(f"Field path '{field_path}' not found")
            return current
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")


@tool(
    name="list_sorter",
    description="Sort a list of items",
    timeout=5.0
)
def list_sorter(items: List[str], reverse: bool = False) -> List[str]:
    """
    Sort a list of items.
    
    Args:
        items: List of items to sort
        reverse: Whether to sort in reverse order (default: False)
        
    Returns:
        Sorted list
    """
    try:
        return sorted(items, reverse=reverse)
    except Exception as e:
        raise ValueError(f"Error sorting list: {str(e)}")


# Utility tools
@tool(
    name="sleep",
    description="Sleep for specified number of seconds",
    timeout=30.0
)
async def sleep_tool(seconds: float) -> str:
    """
    Sleep for specified number of seconds.
    
    Args:
        seconds: Number of seconds to sleep
        
    Returns:
        Confirmation message
    """
    if seconds < 0 or seconds > 10:
        raise ValueError("Sleep duration must be between 0 and 10 seconds")
    
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds"


@tool(
    name="memory_info",
    description="Get information about agent memory usage",
    timeout=2.0
)
def memory_info() -> Dict[str, Any]:
    """
    Get information about current memory usage.
    
    Returns:
        Memory usage information
    """
    import psutil
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


# Mock web search tool
@tool(
    name="web_search",
    description="Search the web for information (mock implementation)",
    timeout=3.0
)
async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Mock web search tool.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Simulate search delay
    await asyncio.sleep(0.5)
    
    # Mock search results
    mock_results = [
        {
            "title": f"Result about {query} - Example Site",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}"
        },
        {
            "title": f"Understanding {query} - Wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        },
        {
            "title": f"How to {query} - Tutorial Site",
            "url": f"https://tutorials.com/{query.replace(' ', '-')}"
        }
    ]
    
    return mock_results[:max_results]


# Create a list of all example tools for easy registration
EXAMPLE_TOOLS = [
    calculator,
    random_number,
    text_analyzer,
    text_transformer,
    current_time,
    timestamp_converter,
    file_reader,
    file_writer,
    directory_lister,
    json_parser,
    list_sorter,
    sleep_tool,
    memory_info,
    web_search
]