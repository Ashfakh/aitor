"""
Centralized logging configuration for Aitor framework.
Provides structured logging for reasoning, planning, and tool execution.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format timestamp
        record.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        return super().format(record)


class AitorLogger:
    """
    Centralized logger for Aitor framework with structured output.
    """
    
    def __init__(self, name: str = "aitor"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = ColoredFormatter(
            fmt='%(timestamp)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def log_prompt(self, component: str, prompt: str, context: Optional[Dict[str, Any]] = None):
        """Log LLM prompt with context."""
        separator = "=" * 80
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🤖 {component.upper()} - LLM PROMPT")
        self.logger.info(f"{separator}")
        
        if context:
            self.logger.info(f"Context: {context}")
            self.logger.info("-" * 40)
        
        self.logger.info(prompt)
        self.logger.info(f"{separator}")
    
    def log_response(self, component: str, response: str, context: Optional[Dict[str, Any]] = None):
        """Log LLM response with context."""
        separator = "=" * 80
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🧠 {component.upper()} - LLM RESPONSE")
        self.logger.info(f"{separator}")
        
        if context:
            self.logger.info(f"Context: {context}")
            self.logger.info("-" * 40)
        
        self.logger.info(response)
        self.logger.info(f"{separator}")
    
    def log_reasoning_step(self, component: str, step_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log reasoning step with details."""
        icon = {
            'THINK': '💭',
            'ACT': '⚡',
            'OBSERVE': '👀',
            'FINAL_ANSWER': '✅',
            'PLAN': '📋',
            'EXECUTE': '🔧',
            'DELEGATE': '👥'
        }.get(step_type, '🔄')
        
        self.logger.info(f"\n{icon} {component.upper()} - {step_type}")
        self.logger.info("-" * 50)
        self.logger.info(f"Content: {content}")
        
        if metadata:
            self.logger.info(f"Metadata: {metadata}")
        self.logger.info("-" * 50)
    
    def log_todo_created(self, todo_id: str, content: str, priority: str):
        """Log todo creation."""
        self.logger.info("\n📝 TODO CREATED")
        self.logger.info("-" * 30)
        self.logger.info(f"ID: {todo_id}")
        self.logger.info(f"Priority: {priority}")
        self.logger.info(f"Content: {content}")
        self.logger.info("-" * 30)
    
    def log_todo_status_change(self, todo_id: str, content: str, old_status: str, new_status: str):
        """Log todo status change."""
        status_icons = {
            'pending': '⏳',
            'in_progress': '🔄',
            'completed': '✅',
            'failed': '❌',
            'cancelled': '🚫'
        }
        
        old_icon = status_icons.get(old_status, '❓')
        new_icon = status_icons.get(new_status, '❓')
        
        self.logger.info("\n📋 TODO STATUS CHANGE")
        self.logger.info("-" * 30)
        self.logger.info(f"ID: {todo_id}")
        self.logger.info(f"Content: {content}")
        self.logger.info(f"Status: {old_icon} {old_status} → {new_icon} {new_status}")
        self.logger.info("-" * 30)
    
    def log_sub_agent_execution(self, agent_name: str, task: str, result: Optional[str] = None, error: Optional[str] = None):
        """Log sub-agent execution."""
        self.logger.info("\n🤖 SUB-AGENT EXECUTION")
        self.logger.info("-" * 40)
        self.logger.info(f"Agent: {agent_name}")
        self.logger.info(f"Task: {task}")
        
        if result:
            self.logger.info(f"Result: {result}")
        if error:
            self.logger.error(f"Error: {error}")
        
        self.logger.info("-" * 40)
    
    def log_tool_execution(self, tool_name: str, params: Dict[str, Any], result: Any, success: bool):
        """Log tool execution."""
        icon = "🔧" if success else "❌"
        
        self.logger.info(f"\n{icon} TOOL EXECUTION")
        self.logger.info("-" * 30)
        self.logger.info(f"Tool: {tool_name}")
        self.logger.info(f"Params: {params}")
        self.logger.info(f"Success: {success}")
        self.logger.info(f"Result: {result}")
        self.logger.info("-" * 30)
    
    def log_planning_summary(self, total_todos: int, completed: int, failed: int, pending: int):
        """Log planning progress summary."""
        self.logger.info("\n📊 PLANNING SUMMARY")
        self.logger.info("-" * 30)
        self.logger.info(f"Total Todos: {total_todos}")
        self.logger.info(f"Completed: {completed}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Pending: {pending}")
        self.logger.info(f"Progress: {(completed/total_todos*100):.1f}%" if total_todos > 0 else "Progress: 0%")
        self.logger.info("-" * 30)
    
    def log_section_start(self, section_name: str):
        """Log the start of a major section."""
        separator = "=" * 80
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🚀 {section_name.upper()}")
        self.logger.info(f"{separator}")
    
    def log_section_end(self, section_name: str):
        """Log the end of a major section."""
        separator = "=" * 80
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🎉 {section_name.upper()} COMPLETED")
        self.logger.info(f"{separator}")


# Global logger instance
aitor_logger = AitorLogger()


def setup_aitor_logging(level: str = "INFO"):
    """
    Setup Aitor logging with specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set specific loggers
    for logger_name in ['aitor', 'aitor.reasoning', 'aitor.planning_reasoning', 'aitor.tools', 'aitor.sub_agent']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"aitor.{name}")


# Convenience functions for structured logging
def log_prompt(component: str, prompt: str, context: Optional[Dict[str, Any]] = None):
    """Log LLM prompt."""
    aitor_logger.log_prompt(component, prompt, context)


def log_response(component: str, response: str, context: Optional[Dict[str, Any]] = None):
    """Log LLM response."""
    aitor_logger.log_response(component, response, context)


def log_reasoning_step(component: str, step_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    """Log reasoning step."""
    aitor_logger.log_reasoning_step(component, step_type, content, metadata)


def log_todo_created(todo_id: str, content: str, priority: str):
    """Log todo creation."""
    aitor_logger.log_todo_created(todo_id, content, priority)


def log_todo_status_change(todo_id: str, content: str, old_status: str, new_status: str):
    """Log todo status change."""
    aitor_logger.log_todo_status_change(todo_id, content, old_status, new_status)


def log_sub_agent_execution(agent_name: str, task: str, result: Optional[str] = None, error: Optional[str] = None):
    """Log sub-agent execution."""
    aitor_logger.log_sub_agent_execution(agent_name, task, result, error)


def log_tool_execution(tool_name: str, params: Dict[str, Any], result: Any, success: bool):
    """Log tool execution."""
    aitor_logger.log_tool_execution(tool_name, params, result, success)


def log_planning_summary(total_todos: int, completed: int, failed: int, pending: int):
    """Log planning summary."""
    aitor_logger.log_planning_summary(total_todos, completed, failed, pending)


def log_section_start(section_name: str):
    """Log section start."""
    aitor_logger.log_section_start(section_name)


def log_section_end(section_name: str):
    """Log section end."""
    aitor_logger.log_section_end(section_name)