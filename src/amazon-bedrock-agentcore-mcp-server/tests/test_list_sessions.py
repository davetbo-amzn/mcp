"""Tests for list_sessions functionality in AgentCore MCP server.

This test module validates the list_sessions functionality against real AWS AgentCore services.
Following TDD principles: real service testing, fail-fast methodology, no mocks.
"""

import pytest
import os 
from awslabs.amazon_bedrock_agentcore_mcp_server.memory import AgentCoreMemoryClient, MemoryConfig


def test_list_sessions_basic():
    """Test basic list_sessions functionality with real AWS service."""
    # Get memory configuration from environment
    memory_config = MemoryConfig.from_environment()
    client = AgentCoreMemoryClient(memory_config)

    # Call list_sessions with default parameters
    result = client.list_sessions(max_results=10)

    # Validate response structure
    assert 'sessions' in result
    assert isinstance(result['sessions'], list)

    # Log results for visibility
    sessions = result['sessions']
    print(f"\nFound {len(sessions)} sessions")

    # If sessions exist, validate their structure
    if sessions:
        session = sessions[0]
        assert 'sessionId' in session
        assert 'createdAt' in session
        assert 'actorId' in session

        print(f"Sample session ID: {session['sessionId']}")
        print(f"Created at: {session['createdAt']}")
        print(f"Actor ID: {session['actorId']}")


def test_list_sessions_with_custom_max_results():
    """Test list_sessions with custom max_results parameter."""
    memory_config = MemoryConfig.from_environment()
    client = AgentCoreMemoryClient(memory_config)

    # Test with different max_results values
    for max_results in [5, 15, 25]:
        result = client.list_sessions(max_results=max_results)

        assert 'sessions' in result
        sessions = result['sessions']

        # Verify we don't exceed max_results
        assert len(sessions) <= max_results

        print(f"\nWith max_results={max_results}: found {len(sessions)} sessions")


def test_list_sessions_response_format():
    """Test that list_sessions returns properly formatted session data."""
    memory_config = MemoryConfig.from_environment()
    client = AgentCoreMemoryClient(memory_config)

    result = client.list_sessions(max_results=10)

    # Validate top-level response structure
    assert isinstance(result, dict)
    assert 'sessions' in result

    sessions = result['sessions']

    # If sessions exist, validate each session's structure
    for session in sessions:
        # Required fields
        assert 'sessionId' in session, "Session must have a 'sessionId' field"
        assert isinstance(session['sessionId'], str), "Session ID must be a string"

        # Required fields from AWS API
        assert 'actorId' in session, "Session must have an 'actorId' field"
        assert isinstance(session['actorId'], str), "Actor ID must be a string"

        assert 'createdAt' in session, "Session must have a 'createdAt' field"

        print(f"\nValidated session: {session['sessionId']}")


if __name__ == '__main__':
    pytest.main([__file__, '-xsv'])
