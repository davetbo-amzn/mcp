"""Memory module for AgentCore MCP Server.

This module provides memory management functionality for the AgentCore MCP Server,
including memory event operations, memory record operations, and memory strategy management.
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .client import AgentCoreMemoryClient
from .config import MemoryConfig
from .memory_strategy_config import MemoryStrategyConfig
from .types import MemoryEvent, ConversationMessage, MemoryRecord

__all__ = [
    "AgentCoreMemoryClient",
    "MemoryConfig", 
    "MemoryStrategyConfig",
    "MemoryEvent",
    "ConversationMessage",
    "MemoryRecord",
]