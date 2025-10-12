"""Integration tests for memory tools in the MCP server.

This module tests the integration of memory functionality with the main server,
including tool registration, error handling, and end-to-end workflows.
Following TDD principles: real service testing, fail-fast methodology, no mocks.
"""

import os
import logging

import pytest

# Import the server module and memory components
from awslabs.amazon_bedrock_agentcore_mcp_server.server import g
    create_memory_event,
    list_memory_events,
    retrieve_memories,
    list_actors,
    delete_event
)
from awslabs.amazon_bedrock_agentcore_mcp_server.memory import MemoryConfig

logger = logging.getLogger(__name__)


class TestMemoryToolsIntegration:
    """Test memory tools integration with the MCP server using real AWS services."""

    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run integration tests"
    )
    def test_create_memory_event_success(self):
        """Test successful memory event creation with real service."""
        test_messages = [
            ['Hello, integration test message', 'USER'],
            ['I received your integration test message!', 'ASSISTANT']
        ]

        result = create_memory_event(test_messages)

        assert result['message'] == 'Memory event created successfully'
        assert 'session_id' in result
        assert 'event_id' in result
        assert 'actor_id' in result
        assert isinstance(result['session_id'], str)
        assert isinstance(result['event_id'], str)
        assert isinstance(result['actor_id'], str)
        logger.info("Created memory event: %s", result['event_id'])

    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run integration tests"
    )
    def test_create_memory_event_with_explicit_session(self):
        """Test memory event creation with explicit session ID."""
        test_messages = [
            ['Test with session ID', 'USER'],
            ['Acknowledged', 'ASSISTANT']
        ]
        session_id = "integration-test-session"

        result = create_memory_event(test_messages, session_id=session_id)

        assert result['message'] == 'Memory event created successfully'
        assert result['session_id'] == session_id
        assert 'event_id' in result
        logger.info("Created event in session: %s", session_id)

    def test_create_memory_event_invalid_message_format(self):
        """Test memory event creation with invalid message format."""
        invalid_messages = [
            ['Hello'],  # Missing role
        ]

        with pytest.raises(RuntimeError, match="Each message must be a \\[content, role\\] tuple"):
            create_memory_event(invalid_messages)

    def test_create_memory_event_invalid_role(self):
        """Test memory event creation with invalid role."""
        invalid_messages = [
            ['Hello', 'INVALID_ROLE']
        ]

        with pytest.raises(RuntimeError, match="Invalid role 'INVALID_ROLE'"):
            create_memory_event(invalid_messages)

    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run integration tests"
    )
    def test_list_memory_events_success(self):
        """Test successful memory events listing with real service."""
        # First create an event to ensure we have something to list
        test_messages = [
            ['List test message', 'USER'],
            ['Response to list test', 'ASSISTANT']
        ]
        session_id = "integration-list-test-session"
        create_memory_event(test_messages, session_id=session_id)

        # Now list events for that session
        result = list_memory_events(session_id)

        assert 'message' in result
        assert 'count' in result
        assert 'events' in result
        assert isinstance(result['events'], list)
        assert result['count'] >= 1

        # Validate event structure
        if result['events']:
            event = result['events'][0]
            assert 'event_id' in event
            assert 'session_id' in event
            assert event['session_id'] == session_id
            assert 'messages' in event
            assert isinstance(event['messages'], list)
            logger.info("Listed %d events for session", result['count'])

    @pytest.mark.skipif(
        not any([
            os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'),
            os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'),
            os.getenv('MEMORY_STRATEGY_ID_SUMMARIZATION')
        ]),
        reason="No memory strategies configured - set MEMORY_STRATEGY_ID_* environment variables"
    )
    def test_retrieve_memories_success(self):
        """Test successful memory retrieval with real service."""
        test_query = 'integration test query'

        # Test with first available strategy
        if os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'):
            strategy = 'semantic'
        elif os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'):
            strategy = 'user_preferences'
        else:
            strategy = 'summarization'

        result = retrieve_memories(test_query, strategy)

        assert 'message' in result
        assert 'count' in result
        assert 'strategy_type' in result
        assert 'memories' in result
        assert result['strategy_type'] == strategy
        assert isinstance(result['memories'], list)
        logger.info("Retrieved %d memories using %s strategy", result['count'], strategy)

        # Validate memory structure if any exist
        if result['memories']:
            memory = result['memories'][0]
            assert 'memory_record_id' in memory
            assert 'content' in memory
            assert 'relevance_score' in memory
            assert isinstance(memory['relevance_score'], float)

    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run integration tests"
    )
    def test_list_actors_success(self):
        """Test successful actors listing with real service."""
        result = list_actors(max_results=5)

        assert 'message' in result
        assert 'count' in result
        assert 'actors' in result
        assert isinstance(result['actors'], list)
        logger.info("Listed %d actors", result['count'])

        # Validate actor structure if any exist
        if result['actors']:
            actor = result['actors'][0]
            assert isinstance(actor, dict)

    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run integration tests"
    )
    def test_delete_event_success(self):
        """Test successful event deletion with real service."""
        # First create an event to delete
        test_messages = [
            ['Delete test message', 'USER'],
            ['This will be deleted', 'ASSISTANT']
        ]
        session_id = "integration-delete-test-session"
        create_result = create_memory_event(test_messages, session_id=session_id)

        event_id = create_result['event_id']

        # Now delete it
        result = delete_event(event_id, session_id)

        assert result['message'] == 'Memory event deleted successfully'
        assert result['event_id'] == event_id
        logger.info("Deleted event: %s", event_id)


class TestMemoryToolsEndToEnd:
    """End-to-end tests for memory tools workflow using real AWS services."""

    @pytest.mark.skipif(
        not all([
            os.getenv('AGENTCORE_MCP_MEMORY_ID'),
            any([
                os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'),
                os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'),
                os.getenv('MEMORY_STRATEGY_ID_SUMMARIZATION')
            ])
        ]),
        reason="Memory ID and at least one strategy required for end-to-end tests"
    )
    def test_complete_memory_workflow(self):
        """Test complete memory workflow from creation to retrieval with real services."""
        # Determine which strategy to use
        if os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'):
            strategy = 'semantic'
        elif os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'):
            strategy = 'user_preferences'
        else:
            strategy = 'summarization'

        # Step 1: Create memory event
        test_messages = [
            ['End-to-end workflow test message', 'USER'],
            ['Processing your workflow test', 'ASSISTANT']
        ]
        session_id = "e2e-workflow-session"

        create_result = create_memory_event(test_messages, session_id=session_id)
        assert create_result['message'] == 'Memory event created successfully'
        assert create_result['session_id'] == session_id
        assert 'event_id' in create_result
        logger.info("Step 1: Created event %s", create_result['event_id'])

        # Step 2: List memory events
        list_result = list_memory_events(session_id)
        assert list_result['count'] >= 1
        assert any(e['event_id'] == create_result['event_id'] for e in list_result['events'])
        logger.info("Step 2: Listed %d events", list_result['count'])

        # Step 3: Retrieve memories
        retrieve_result = retrieve_memories('workflow test', strategy)
        assert 'memories' in retrieve_result
        assert retrieve_result['strategy_type'] == strategy
        logger.info("Step 3: Retrieved %d memories", retrieve_result['count'])

        # Step 4: List actors
        actors_result = list_actors()
        assert actors_result['count'] >= 1
        logger.info("Step 4: Listed %d actors", actors_result['count'])

        logger.info("End-to-end workflow completed successfully")


class TestBackwardCompatibility:
    """Test backward compatibility with existing documentation tools."""

    def test_documentation_tools_still_work(self):
        """Test that existing documentation tools are not affected by memory integration."""
        import inspect
        from awslabs.amazon_bedrock_agentcore_mcp_server.server import (
            search_agentcore_docs,
            fetch_agentcore_doc
        )

        # These should be importable and callable
        assert callable(search_agentcore_docs)
        assert callable(fetch_agentcore_doc)

        # Test that the function signatures haven't changed
        search_sig = inspect.signature(search_agentcore_docs)
        assert 'query' in search_sig.parameters
        assert 'k' in search_sig.parameters

        fetch_sig = inspect.signature(fetch_agentcore_doc)
        assert 'uri' in fetch_sig.parameters

    def test_server_starts_without_memory_config(self):
        """Test that server can start without memory configuration."""
        # Save original environment
        memory_vars = [
            'AGENTCORE_MCP_MEMORY_ID',
            'MEMORY_STRATEGY_ID_USER_PREFERENCES',
            'MEMORY_STRATEGY_ID_SEMANTIC',
            'MEMORY_STRATEGY_ID_SUMMARIZATION'
        ]
        original_env = {var: os.environ.get(var) for var in memory_vars}

        # Temporarily remove memory environment variables
        for var in memory_vars:
            if var in os.environ:
                del os.environ[var]

        try:
            # Test memory config loading without configuration
            config = MemoryConfig.from_environment()
            assert config.memory_id is None
            assert not config.has_memory_strategies()
            assert not config.get_configured_strategies()
        finally:
            # Restore original environment
            for var, value in original_env.items():
                if value is not None:
                    os.environ[var] = value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])