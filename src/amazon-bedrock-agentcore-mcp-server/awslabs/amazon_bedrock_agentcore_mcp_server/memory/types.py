"""Type definitions for memory functionality.

This module provides data classes and type definitions used throughout
the memory management system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Literal


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