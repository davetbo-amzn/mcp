"""Type definitions for memory functionality.

This module provides data classes and type definitions used throughout
the memory management system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation.
    
    Attributes:
        role: The role of the message sender (USER, ASSISTANT, or TOOL)
        content: The content of the message
    """
    role: Literal['USER', 'ASSISTANT', 'TOOL']
    content: str


@dataclass
class MemoryEvent:
    """Represents a memory event containing conversation messages.
    
    Attributes:
        event_id: Unique identifier for the event
        session_id: Session identifier the event belongs to
        actor_id: Actor (user) identifier
        timestamp: When the event was created
        messages: List of conversation messages in the event
    """
    event_id: str
    session_id: str
    actor_id: str
    timestamp: datetime
    messages: List[ConversationMessage]


@dataclass
class MemoryRecord:
    """Represents a memory record retrieved from semantic search.
    
    Attributes:
        memory_record_id: Unique identifier for the memory record
        content: The content of the memory record
        timestamp: When the memory record was created
        relevance_score: Relevance score from semantic search
        namespace: Optional namespace for the memory record
    """
    memory_record_id: str
    content: str
    timestamp: datetime
    relevance_score: float
    namespace: Optional[str] = None


@dataclass
class Memory:
    """Represents an AgentCore memory resource.
    
    Attributes:
        memory_id: Unique identifier for the memory resource
        memory_name: Human-readable name for the memory
        memory_arn: Amazon Resource Name for the memory
        strategies: List of memory strategy configurations
        created_at: When the memory resource was created
    """
    memory_id: str
    memory_name: str
    memory_arn: str
    strategies: List[Dict[str, Any]]
    created_at: datetime


@dataclass
class MemoryStrategy:
    """Represents a memory strategy configuration.
    
    Attributes:
        strategy_id: Unique identifier for the strategy
        strategy_name: Human-readable name for the strategy
        strategy_type: Type of strategy (SEMANTIC, SUMMARIZATION, USER_PREFERENCES)
        namespaces: List of namespaces this strategy operates in
    """
    strategy_id: str
    strategy_name: str
    strategy_type: str
    namespaces: List[str]


class MemoryProviderSettings(BaseModel):
    """Settings for a single memory provider.
    
    Attributes:
        memory_id: Unique identifier for the memory resource
        strategies: Dictionary mapping strategy types to strategy IDs
        created_at: ISO 8601 timestamp when the memory was created
    """
    memory_id: str = Field(description="Memory resource ID")
    strategies: Dict[str, str] = Field(
        description="Map of strategy type to strategy ID",
        default_factory=dict
    )
    created_at: str = Field(description="ISO 8601 timestamp of creation")


class MemorySettings(BaseModel):
    """Schema for memory_settings.json file.
    
    Supports multiple memory providers keyed by memory name.
    
    Attributes:
        disable_memory: Flag to disable memory prompts if user declined
        memory_providers: Dictionary of memory providers keyed by memory name
    """
    disable_memory: bool = Field(
        default=False,
        description="If true, user declined memory creation and should not be prompted again"
    )
    memory_providers: Dict[str, MemoryProviderSettings] = Field(
        default_factory=dict,
        description="Memory providers keyed by memory name"
    )
