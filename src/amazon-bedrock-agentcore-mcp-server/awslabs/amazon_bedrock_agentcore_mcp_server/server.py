# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""awslabs AWS Bedrock AgentCore MCP Server implementation."""

from .utils.caller_identity import get_actor_id
from .utils import cache, text_processor
from .memory import AgentCoreMemoryClient, MemoryConfig, ConversationMessage
from .memory.types import MemoryEvent, MemoryRecord
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

APP_NAME = 'amazon-bedrock-agentcore-mcp-server'
mcp = FastMCP(APP_NAME)

# Global memory client - initialized on first use
_memory_client: Optional[AgentCoreMemoryClient] = None


def _get_memory_client() -> AgentCoreMemoryClient:
    """Get or create the memory client instance.
    
    Returns:
        AgentCoreMemoryClient instance
        
    Raises:
        RuntimeError: If memory configuration is invalid
    """
    global _memory_client
    
    if _memory_client is None:
        try:
            memory_config = MemoryConfig.from_environment()
            _memory_client = AgentCoreMemoryClient(memory_config)
            
            # Validate configuration
            validation = _memory_client.validate_memory_configuration()
            if not validation['valid']:
                error_msg = f"Memory configuration invalid: {', '.join(validation['errors'])}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"Memory configuration warning: {warning}")
                    
            logger.info("Memory client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory client: {e}")
            raise RuntimeError(f"Memory functionality unavailable: {e}")
    
    return _memory_client


@mcp.tool()
def search_agentcore_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search curated AgentCore documentation and return ranked results with snippets.

    This tool provides access to the complete Amazon Bedrock AgentCore documentation including:

    **Platform Overview:**
    - What is Bedrock AgentCore, security overview, quotas and limits

    **Platform Services:**
    - AgentCore Runtime (serverless deployment and scaling)
    - AgentCore Memory (persistent knowledge with event and semantic memory)
    - AgentCore Code Interpreter (secure code execution in isolated sandboxes)
    - AgentCore Browser (fast, secure cloud-based browser for web interaction)
    - AgentCore Gateway (transform existing APIs into agent tools)
    - AgentCore Observability (real-time monitoring and tracing)
    - AgentCore Identity (secure authentication and access management)

    **Getting Started:**
    - Prerequisites & environment setup
    - Building your first agent or transforming existing code
    - Local development & testing
    - Deployment to AgentCore using CLI
    - Troubleshooting & enhancement

    **Examples & Tutorials:**
    - Basic agent creation, memory integration, tool usage
    - Streaming responses, error handling, authentication
    - Customer service agents, code review assistants, data analysis
    - Multi-agent workflows and integrations

    **API Reference:**
    - Data plane and control API documentation

    Use this to find relevant AgentCore documentation for any development question.

    Args:
        query: Search query string (e.g., "bedrock agentcore", "memory integration", "deployment guide")
        k: Maximum number of results to return (default: 5)

    Returns:
        List of dictionaries containing:
        - url: Document URL
        - title: Display title
        - score: Relevance score (0-1, higher is better)
        - snippet: Contextual content preview

    """
    cache.ensure_ready()
    index = cache.get_index()
    results = index.search(query, k=k) if index else []
    url_cache = cache.get_url_cache()

    # Collect top-k URLs that need hydration (no content yet)
    # Simplified: Direct hydration in one pass
    top = results[: min(len(results), cache.SNIPPET_HYDRATE_MAX)]
    for _, doc in top:
        cached = url_cache.get(doc.uri)
        if cached is None or not cached.content:
            cache.ensure_page(doc.uri)

    # Build response with real content snippets when available
    return_docs: List[Dict[str, Any]] = []
    for score, doc in results:
        page = url_cache.get(doc.uri)
        snippet = text_processor.make_snippet(page, doc.display_title)
        return_docs.append(
            {
                'url': doc.uri,
                'title': doc.display_title,
                'score': round(score, 3),
                'snippet': snippet,
            }
        )
    return return_docs


@mcp.tool()
def fetch_agentcore_doc(uri: str) -> Dict[str, Any]:
    """Fetch full document content by URL.

    Retrieves complete AgentCore documentation content from URLs found via search_agentcore_docs
    or provided directly. Use this to get full documentation pages including:

    - Complete platform overview and service documentation
    - Detailed getting started guides with step-by-step instructions
    - Full API reference documentation
    - Comprehensive tutorial and example code
    - Complete deployment and configuration instructions
    - Integration guides for various frameworks (Strands, LangGraph, CrewAI, etc.)

    This provides the full content when search snippets aren't sufficient for
    understanding or implementing AgentCore features.

    Args:
        uri: Document URI (supports http/https URLs)

    Returns:
        Dictionary containing:
        - url: Canonical document URL
        - title: Document title
        - content: Full document text content
        - error: Error message (if fetch failed)

    """
    cache.ensure_ready()

    page = cache.ensure_page(uri)
    if page is None:
        return {'error': 'fetch failed', 'url': uri}

    return {
        'url': page.url,
        'title': page.title,
        'content': page.content,
    }


@mcp.tool()
def create_memory_event(
    messages: List[List[str]], 
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a memory event with conversation messages.
    
    This tool creates a new memory event in AgentCore Memory containing conversation
    messages. Memory events are used to store conversational context that can be
    retrieved later for personalized agent experiences.
    
    Args:
        messages: List of [content, role] tuples. Roles must be USER, ASSISTANT, or TOOL
        session_id: Optional. Unique session identifier. Should be omitted on the first
             call in a session and the server will create the session_id and return it 
             for use in future calls in the same session. 

    Returns:
        Dictionary containing:
        - message: Success message
        - session_id: Session ID for the created event
        - event_id: ID of the created event
        - actor_id: Actor ID associated with the event
        
    Raises:
        RuntimeError: If memory is not configured or event creation fails
    """
    try:
        client = _get_memory_client()
        
        # Convert message tuples to ConversationMessage objects
        conversation_messages = []
        for message_data in messages:
            if len(message_data) != 2:
                raise ValueError("Each message must be a [content, role] tuple")
            
            content, role = message_data
            if role not in ['USER', 'ASSISTANT', 'TOOL']:
                raise ValueError(f"Invalid role '{role}'. Must be USER, ASSISTANT, or TOOL")
            
            conversation_messages.append(ConversationMessage(role=role, content=content))
        
        # Create the memory event
        response = client.create_memory_event(conversation_messages, session_id)['event']
        
        return {
            'message': 'Memory event created successfully',
            'session_id': response.get('sessionId', session_id),
            'event_id': response.get('eventId'),
            'actor_id': client.actor_id
        }
        
    except Exception as e:
        logger.error(f"Failed to create memory event: {e}")
        raise RuntimeError(f"Failed to create memory event: {e}")


@mcp.tool()
def list_memory_events(
    session_id: str,
    max_results: int = 15
) -> Dict[str, Any]:
    """List memory events with optional filtering.
    
    Retrieves memory events for a specific session, providing access to stored
    conversation history and context.
    
    Args:
        max_results: Maximum number of results (default 15)
        session_id: the session identifier created earlier in the thread after a create_event.
        
    Returns:
        Dictionary containing:
        - message: Success message
        - events: List of memory events with their details
        - count: Number of events returned
        
    Raises:
        RuntimeError: If memory is not configured or listing fails
    """
    try:
        client = _get_memory_client()
        
        events = client.list_memory_events(session_id, max_results)
        
        # Convert MemoryEvent objects to dictionaries
        event_dicts = []
        for event in events:
            event_dict = {
                'event_id': event.event_id,
                'session_id': event.session_id,
                'actor_id': event.actor_id,
                'timestamp': event.timestamp.isoformat(),
                'messages': [
                    {'role': msg.role, 'content': msg.content}
                    for msg in event.messages
                ]
            }
            event_dicts.append(event_dict)
        
        return {
            'message': f'Retrieved {len(events)} memory events',
            'events': event_dicts,
            'count': len(events)
        }
        
    except Exception as e:
        logger.error(f"Failed to list memory events: {e}")
        raise RuntimeError(f"Failed to list memory events: {e}")


@mcp.tool()
def list_sessions(max_results: int = 10) -> Dict[str, Any]:
    """List conversation sessions for the current AWS caller identity.

    Retrieves information about conversation sessions, providing insights into
    active and past conversations.

    Args:
        max_results: Maximum number of results (default: 10)

    Returns:
        Dictionary containing:
        - message: Success message
        - sessions: List of session information
        - count: Number of sessions returned

    Raises:
        RuntimeError: If memory is not configured or listing fails

    Note: Sessions are automatically filtered by the AWS STS caller identity ARN.
    """
    try:
        client = _get_memory_client()

        # list_sessions returns a list directly
        sessions = client.list_sessions(max_results)

        # Format response
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                'session_id': session.get('sessionId'),
                'created_at': session.get('createdAt'),
                'actor_id': session.get('actorId')
            })

        return {
            'message': f'Retrieved {len(formatted_sessions)} sessions',
            'sessions': formatted_sessions
        }

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise RuntimeError(f"Failed to list sessions: {e}")


@mcp.tool()
def retrieve_memories(
    query: str,
    memory_strategy_type: str,
    session_id: Optional[str] = None,
    max_results: int = 15
) -> Dict[str, Any]:
    """Retrieve memory records using semantic search.
    
    Searches through stored memories using the specified strategy type to find
    relevant content based on the query. This enables agents to access historical
    context, user preferences, and accumulated knowledge.
    
    Args:
        query: Query string for memory retrieval
        memory_strategy_type: The memory strategy type to use. Supported types: user_preferences, summarization, semantic
        session_id: If included it will be used to retrieve memories from a specific session. If omitted it will search across all sessions.
        max_results: Maximum number of results (default: 15)
        
    Returns:
        Dictionary containing:
        - message: Success message
        - memories: List of retrieved memory records
        - count: Number of memories returned
        - strategy_type: Strategy type used for retrieval
        
    Raises:
        RuntimeError: If memory is not configured or retrieval fails
    """
    try:
        client = _get_memory_client()
        
        records = client.retrieve_memories(query, memory_strategy_type, session_id, max_results)
        
        # Convert MemoryRecord objects to dictionaries
        record_dicts = []
        for record in records:
            record_dict = {
                'memory_record_id': record.memory_record_id,
                'content': record.content,
                'timestamp': record.timestamp.isoformat(),
                'relevance_score': record.relevance_score,
                'namespace': record.namespace
            }
            record_dicts.append(record_dict)
        
        return {
            'message': f'Retrieved {len(records)} memory records',
            'memories': record_dicts,
            'count': len(records),
            'strategy_type': memory_strategy_type
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        raise RuntimeError(f"Failed to retrieve memories: {e}")


@mcp.tool()
def list_actors(max_results: int = 10) -> Dict[str, Any]:
    """List actors (participants) in memory conversations.
    
    Retrieves information about actors who have participated in memory conversations,
    providing insights into conversation participants and their activity.
    
    Args:
        max_results: Maximum number of results (default: 10)
        
    Returns:
        Dictionary containing:
        - message: Success message
        - actors: List of actor information
        - count: Number of actors returned
        
    Raises:
        RuntimeError: If memory is not configured or listing fails
    """
    try:
        client = _get_memory_client()
        
        actors = client.list_actors(max_results)
        
        return {
            'message': f'Retrieved {len(actors)} actors',
            'actors': actors,
            'count': len(actors)
        }
        
    except Exception as e:
        logger.error(f"Failed to list actors: {e}")
        raise RuntimeError(f"Failed to list actors: {e}")


@mcp.tool()
def delete_event(event_id: str, session_id: str) -> Dict[str, Any]:
    """Delete a specific memory event.
    
    Removes a memory event from storage. This is useful for cleaning up
    unwanted or erroneous conversation history.
    
    Args:
        event_id: Event ID to delete
        session_id: session ID in which the event was created.
        
    Returns:
        Dictionary containing:
        - message: Success message
        - event_id: ID of the deleted event
        
    Raises:
        RuntimeError: If memory is not configured or deletion fails
    """
    try:
        client = _get_memory_client()
        
        client.delete_event(event_id, session_id)
        
        return {
            'message': 'Memory event deleted successfully',
            'event_id': event_id
        }
        
    except Exception as e:
        logger.error(f"Failed to delete memory event: {e}")
        raise RuntimeError(f"Failed to delete memory event: {e}")


@mcp.tool()
def delete_memory_record(memory_record_id: str) -> Dict[str, Any]:
    """Delete a specific memory record.
    
    Removes a memory record from storage. This is useful for cleaning up
    outdated or incorrect memory content.
    
    Args:
        memory_record_id: id of the memory record
        
    Returns:
        Dictionary containing:
        - message: Success message
        - memory_record_id: ID of the deleted memory record
        
    Raises:
        RuntimeError: If memory is not configured or deletion fails
    """
    try:
        client = _get_memory_client()
        
        client.delete_memory_record(memory_record_id)
        
        return {
            'message': 'Memory record deleted successfully',
            'memory_record_id': memory_record_id
        }
        
    except Exception as e:
        logger.error(f"Failed to delete memory record: {e}")
        raise RuntimeError(f"Failed to delete memory record: {e}")


def main() -> None:
    """Main entry point for the MCP server.

    Initializes the document cache, memory configuration, and starts the FastMCP server.
    The cache is loaded with document titles only for fast startup,
    with full content fetched on-demand.
    
    Memory functionality is initialized with graceful degradation - if memory
    configuration is missing or invalid, the server will still start but
    memory tools will return appropriate error messages.
    """
    # Initialize documentation cache
    cache.ensure_ready()
    
    # Initialize memory configuration with graceful degradation
    try:
        memory_config = MemoryConfig.from_environment()
        
        if memory_config.memory_id:
            # Try to initialize memory client to validate configuration
            client = AgentCoreMemoryClient(memory_config)
            validation = client.validate_memory_configuration()
            
            if validation['valid']:
                logger.info("Memory functionality initialized successfully")
                if validation['warnings']:
                    for warning in validation['warnings']:
                        logger.warning(f"Memory configuration: {warning}")
                        
                # Log available strategies
                configured_strategies = validation['strategies_configured']
                if configured_strategies:
                    logger.info(f"Configured memory strategies: {', '.join(configured_strategies)}")
                else:
                    logger.warning("No memory strategies configured")
                    
            else:
                logger.error(f"Memory configuration invalid: {', '.join(validation['errors'])}")
                logger.warning("Memory tools will be unavailable")
        else:
            logger.info("Memory functionality not configured (AGENTCORE_MCP_MEMORY_ID not set)")
            logger.info("Memory tools will be unavailable")
            
    except Exception as e:
        logger.warning(f"Memory initialization failed: {e}")
        logger.info("Memory tools will be unavailable")
    
    # Start the MCP server
    logger.info("Starting AgentCore MCP Server with documentation and memory tools")
    mcp.run()


if __name__ == '__main__':
    main()
