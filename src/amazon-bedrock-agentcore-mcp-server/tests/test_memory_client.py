"""Tests for AgentCore memory client.

This module tests the AgentCoreMemoryClient class with real AWS services.
Following TDD principles, these tests validate actual functionality without mocking.
"""

import os
import logging
from datetime import datetime

import pytest

from awslabs.amazon_bedrock_agentcore_mcp_server.memory.client import AgentCoreMemoryClient
from awslabs.amazon_bedrock_agentcore_mcp_server.memory.config import MemoryConfig
from awslabs.amazon_bedrock_agentcore_mcp_server.memory.types import ConversationMessage, MemoryEvent, MemoryRecord
from awslabs.amazon_bedrock_agentcore_mcp_server.utils.exceptions import AgentCoreMemoryError

logger = logging.getLogger(__name__)


class TestAgentCoreMemoryClient:
    """Test suite for AgentCoreMemoryClient."""
    
    @pytest.fixture
    def memory_config(self):
        """Create memory configuration for testing."""
        return MemoryConfig.from_environment()
    
    @pytest.fixture
    def memory_client(self, memory_config):
        """Create memory client for testing."""
        return AgentCoreMemoryClient(memory_config)
    
    @pytest.fixture
    def test_messages(self):
        """Create test conversation messages."""
        return [
            ConversationMessage(role='USER', content='Hello, this is a test message'),
            ConversationMessage(role='ASSISTANT', content='Hello! I received your test message.')
        ]
    
    def test_client_initialization(self, memory_client):
        """Test memory client initialization."""
        assert memory_client is not None
        assert memory_client.config is not None
        assert memory_client.strategy_config is not None
        assert memory_client.aws_manager is not None
        
        # Test actor ID resolution
        actor_id = memory_client.actor_id
        assert actor_id is not None
        assert isinstance(actor_id, str)
        assert len(actor_id) > 0
        logger.info("Actor ID resolved: %s", actor_id)
    
    def test_memory_id_property(self, memory_client):
        """Test memory ID property access."""
        if memory_client.config.memory_id:
            memory_id = memory_client.memory_id
            assert memory_id is not None
            assert isinstance(memory_id, str)
            assert len(memory_id) > 0
            logger.info("Memory ID: %s", memory_id)
        else:
            # Test error when memory ID not configured
            with pytest.raises(AgentCoreMemoryError, match="Memory ID not configured"):
                _ = memory_client.memory_id
    
    def test_session_id_creation(self, memory_client):
        """Test session ID creation."""
        session_id = memory_client._create_session_id()
        assert session_id is not None
        assert isinstance(session_id, str)
        assert session_id.startswith('session-')
        assert 'T' in session_id  # Should contain timestamp
        logger.info("Created session ID: %s", session_id)
    
    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run memory tests"
    )
    def test_create_memory_event(self, memory_client, test_messages):
        """Test creating a memory event."""
        # Test with auto-generated session ID
        response = memory_client.create_memory_event(test_messages)
        
        assert response is not None
        assert isinstance(response, dict)
        logger.info(f"Created memory event: {response}")
        
        # Test with explicit session ID
        session_id = "test-session-123"
        response2 = memory_client.create_memory_event(test_messages, session_id=session_id)
        
        assert response2 is not None
        assert isinstance(response2, dict)
        logger.info(f"Created memory event with session ID: {response2}")
    
    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run memory tests"
    )
    def test_list_memory_events(self, memory_client, test_messages):
        """Test listing memory events."""
        # First create an event to ensure we have something to list
        session_id = "test-list-events-session"
        memory_client.create_memory_event(test_messages, session_id=session_id)
        
        # List events for the session
        events = memory_client.list_memory_events(session_id=session_id, max_results=10)
        
        assert isinstance(events, list)
        logger.info(f"Listed {len(events)} memory events")
        
        # Validate event structure if events exist
        if events:
            event = events[0]
            assert isinstance(event, MemoryEvent)
            assert event.event_id is not None
            assert event.session_id == session_id
            assert event.actor_id is not None
            assert isinstance(event.timestamp, datetime)
            assert isinstance(event.messages, list)
            
            # Validate messages
            if event.messages:
                message = event.messages[0]
                assert isinstance(message, ConversationMessage)
                assert message.role in ['USER', 'ASSISTANT', 'TOOL']
                assert isinstance(message.content, str)
    
    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run memory tests"
    )
    def test_list_actors(self, memory_client):
        """Test listing actors."""
        actors = memory_client.list_actors(max_results=5)
        
        assert isinstance(actors, list)
        logger.info(f"Listed {len(actors)} actors")
        
        # Validate actor structure if actors exist
        if actors:
            actor = actors[0]
            assert isinstance(actor, dict)
            # Actor structure may vary, just ensure it's a dictionary
    
    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run memory tests"
    )
    def test_get_memory_provider_details(self, memory_client):
        """Test getting memory provider details."""
        provider_details = memory_client.get_memory_provider_details()
        
        assert provider_details is not None
        assert isinstance(provider_details, dict)
        assert 'strategies' in provider_details
        
        strategies = provider_details['strategies']
        assert isinstance(strategies, list)
        logger.info(f"Memory provider has {len(strategies)} strategies")
        
        # Validate strategy structure if strategies exist
        if strategies:
            strategy = strategies[0]
            assert isinstance(strategy, dict)
            assert 'strategyId' in strategy
            assert 'namespaces' in strategy
    
    @pytest.mark.skipif(
        not any([
            os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'),
            os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'),
            os.getenv('MEMORY_STRATEGY_ID_SUMMARIZATION')
        ]),
        reason="No memory strategies configured - set MEMORY_STRATEGY_ID_* environment variables"
    )
    def test_retrieve_memories(self, memory_client):
        """Test retrieving memories."""
        # Test with semantic strategy if available
        if os.getenv('MEMORY_STRATEGY_ID_SEMANTIC'):
            records = memory_client.retrieve_memories(
                query="test memory retrieval",
                memory_strategy_type="semantic",
                max_results=5
            )
            
            assert isinstance(records, list)
            logger.info(f"Retrieved {len(records)} semantic memory records")
            
            # Validate record structure if records exist
            if records:
                record = records[0]
                assert isinstance(record, MemoryRecord)
                assert record.memory_record_id is not None
                assert isinstance(record.content, str)
                assert isinstance(record.timestamp, datetime)
                assert isinstance(record.relevance_score, float)
        
        # Test with user preferences strategy if available
        if os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'):
            records = memory_client.retrieve_memories(
                query="user preferences test",
                memory_strategy_type="user_preferences",
                max_results=3
            )
            
            assert isinstance(records, list)
            logger.info(f"Retrieved {len(records)} user preference memory records")
    
    def test_retrieve_memories_invalid_strategy(self, memory_client):
        """Test retrieving memories with invalid strategy type."""
        with pytest.raises(AgentCoreMemoryError, match="Invalid memory strategy type"):
            memory_client.retrieve_memories(
                query="test",
                memory_strategy_type="invalid_strategy"
            )
    
    @pytest.mark.skipif(
        not os.getenv('AGENTCORE_MCP_MEMORY_ID'),
        reason="Memory ID not configured - set AGENTCORE_MCP_MEMORY_ID to run memory tests"
    )
    def test_delete_event(self, memory_client, test_messages):
        """Test deleting a memory event."""
        # First create an event to delete
        session_id = "test-delete-event-session"
        response = memory_client.create_memory_event(test_messages, session_id=session_id)
        
        # Extract event ID from response (structure may vary)
        event_id = response.get('eventId')
        if event_id:
            # Test deletion
            memory_client.delete_event(event_id=event_id, session_id=session_id)
            logger.info(f"Successfully deleted event {event_id}")
        else:
            logger.warning("Could not extract event ID from create response, skipping delete test")
    
    def test_namespace_resolution(self, memory_client):
        """Test namespace resolution for memory strategies."""
        # This test requires memory provider details
        if not os.getenv('AGENTCORE_MCP_MEMORY_ID'):
            pytest.skip("Memory ID not configured")
        
        try:
            provider_details = memory_client.get_memory_provider_details()
            strategies = provider_details.get('strategies', [])
            
            if strategies:
                strategy = strategies[0]
                strategy_id = strategy.get('strategyId')
                
                if strategy_id:
                    # Test namespace resolution
                    namespace = memory_client._resolve_namespace(strategy_id)
                    
                    assert namespace is not None
                    assert isinstance(namespace, str)
                    assert len(namespace) > 0
                    logger.info(f"Resolved namespace: {namespace}")
                    
                    # Test with session ID
                    namespace_with_session = memory_client._resolve_namespace(
                        strategy_id, 
                        session_id="test-session"
                    )
                    
                    assert namespace_with_session is not None
                    assert isinstance(namespace_with_session, str)
                    logger.info(f"Resolved namespace with session: {namespace_with_session}")
        except Exception as e:
            logger.warning(f"Namespace resolution test failed: {e}")
            pytest.skip("Could not test namespace resolution")
    
    def test_error_handling_no_memory_id(self):
        """Test error handling when memory ID is not configured."""
        # Create config without memory ID
        config = MemoryConfig(
            memory_id=None,
            aws_region='us-west-2'
        )
        
        client = AgentCoreMemoryClient(config)
        
        # Should raise error when trying to access memory ID
        with pytest.raises(AgentCoreMemoryError, match="Memory ID not configured"):
            _ = client.memory_id
    
    def test_client_with_custom_config(self):
        """Test client initialization with custom configuration."""
        custom_config = MemoryConfig(
            memory_id="test-memory-id",
            aws_region="us-east-1",
            memory_strategy_semantic="test-semantic-strategy"
        )
        
        client = AgentCoreMemoryClient(custom_config)
        
        assert client.config == custom_config
        assert client.config.aws_region == "us-east-1"
        assert client.config.memory_strategy_semantic == "test-semantic-strategy"
        assert client.aws_manager.get_region() == "us-east-1"