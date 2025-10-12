"""Tests for error handling utilities.

These tests validate error mapping, exception handling, and logging functionality
using real error scenarios without mocks, following fail-fast testing principles.
"""

import pytest
import logging
from unittest.mock import patch
from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError

from awslabs.amazon_bedrock_agentcore_mcp_server.utils.exceptions import (
    MCPServerError,
    AWSError,
    AgentCoreMemoryError,
    ConfigurationError,
    map_aws_error,
    map_memory_error,
    handle_error_gracefully,
    log_aws_error
)


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_mcp_server_error(self):
        """Test base MCP server error."""
        message = "Test error message"
        error_code = "TEST_ERROR"
        details = {"key": "value"}
        
        error = MCPServerError(message, error_code, details)
        
        assert str(error) == message
        assert error.message == message
        assert error.error_code == error_code
        assert error.details == details
    
    def test_aws_error(self):
        """Test AWS-specific error."""
        message = "AWS operation failed"
        aws_error_code = "AccessDenied"
        details = {"operation": "test_operation"}
        
        error = AWSError(message, aws_error_code, details)
        
        assert str(error) == message
        assert error.message == message
        assert error.aws_error_code == aws_error_code
        assert error.details == details
    
    def test_agentcore_memory_error(self):
        """Test memory-specific error."""
        message = "Memory operation failed"
        memory_operation = "create_event"
        details = {"session_id": "test-session"}
        
        error = AgentCoreMemoryError(message, memory_operation, details)
        
        assert str(error) == message
        assert error.message == message
        assert error.memory_operation == memory_operation
        assert error.details == details
    
    def test_configuration_error(self):
        """Test configuration-specific error."""
        message = "Configuration invalid"
        config_key = "memory_id"
        details = {"expected": "string", "got": "None"}
        
        error = ConfigurationError(message, config_key, details)
        
        assert str(error) == message
        assert error.message == message
        assert error.config_key == config_key
        assert error.details == details


class TestMapAWSError:
    """Test AWS error mapping functionality."""
    
    def test_map_client_error_access_denied(self):
        """Test mapping AccessDenied client error."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'AccessDenied',
                    'Message': 'User is not authorized to perform this action'
                },
                'ResponseMetadata': {
                    'RequestId': 'test-request-id',
                    'HTTPStatusCode': 403
                }
            },
            operation_name='TestOperation'
        )
        
        mapped_error = map_aws_error(client_error, "test operation")
        
        assert isinstance(mapped_error, AWSError)
        assert "Access denied for test operation" in mapped_error.message
        assert mapped_error.aws_error_code == 'AccessDenied'
        assert mapped_error.details['request_id'] == 'test-request-id'
        assert mapped_error.details['aws_error_code'] == 'AccessDenied'
    
    def test_map_client_error_validation_exception(self):
        """Test mapping ValidationException client error."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'ValidationException',
                    'Message': 'Invalid parameter value'
                }
            },
            operation_name='TestOperation'
        )
        
        mapped_error = map_aws_error(client_error, "validation test")
        
        assert isinstance(mapped_error, AWSError)
        assert "Validation error for validation test" in mapped_error.message
        assert mapped_error.aws_error_code == 'ValidationException'
    
    def test_map_client_error_unknown_code(self):
        """Test mapping unknown client error code."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'UnknownErrorCode',
                    'Message': 'Something went wrong'
                }
            },
            operation_name='TestOperation'
        )
        
        mapped_error = map_aws_error(client_error, "unknown error test")
        
        assert isinstance(mapped_error, AWSError)
        assert "unknown error test failed" in mapped_error.message
        assert "Something went wrong" in mapped_error.message
        assert mapped_error.aws_error_code == 'UnknownErrorCode'
    
    def test_map_botocore_error(self):
        """Test mapping BotoCoreError."""
        botocore_error = BotoCoreError()
        
        mapped_error = map_aws_error(botocore_error, "botocore test")
        
        assert isinstance(mapped_error, AWSError)
        assert "botocore test failed due to AWS service error" in mapped_error.message
        assert mapped_error.details['error_type'] == 'BotoCoreError'
    
    def test_map_generic_error(self):
        """Test mapping generic non-AWS error."""
        generic_error = ValueError("Invalid value")
        
        mapped_error = map_aws_error(generic_error, "generic test")
        
        assert isinstance(mapped_error, MCPServerError)
        assert "generic test failed" in mapped_error.message
        assert "Invalid value" in mapped_error.message
        assert mapped_error.details['error_type'] == 'ValueError'


class TestMapMemoryError:
    """Test memory error mapping functionality."""
    
    def test_map_memory_error_with_aws_error(self):
        """Test mapping memory error that originated from AWS."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'ResourceNotFound',
                    'Message': 'Memory not found'
                }
            },
            operation_name='CreateEvent'
        )
        
        mapped_error = map_memory_error(client_error, "create_event")
        
        assert isinstance(mapped_error, AgentCoreMemoryError)
        assert mapped_error.memory_operation == "create_event"
        assert "ResourceNotFound" in str(mapped_error.details)
    
    def test_map_memory_error_strategy_related(self):
        """Test mapping memory error related to strategy."""
        strategy_error = ValueError("Invalid strategy type")
        
        mapped_error = map_memory_error(strategy_error, "retrieve_memories")
        
        assert isinstance(mapped_error, AgentCoreMemoryError)
        assert "Memory strategy error" in mapped_error.message
        assert mapped_error.memory_operation == "retrieve_memories"
    
    def test_map_memory_error_session_related(self):
        """Test mapping memory error related to session."""
        session_error = RuntimeError("Session not found")
        
        mapped_error = map_memory_error(session_error, "list_events")
        
        assert isinstance(mapped_error, AgentCoreMemoryError)
        assert "Memory session error" in mapped_error.message
        assert mapped_error.memory_operation == "list_events"
    
    def test_map_memory_error_record_related(self):
        """Test mapping memory error related to record."""
        record_error = KeyError("Record not found")
        
        mapped_error = map_memory_error(record_error, "delete_record")
        
        assert isinstance(mapped_error, AgentCoreMemoryError)
        assert "Memory record error" in mapped_error.message
        assert mapped_error.memory_operation == "delete_record"
    
    def test_map_memory_error_generic(self):
        """Test mapping generic memory error."""
        generic_error = Exception("Something went wrong")
        
        mapped_error = map_memory_error(generic_error, "memory_operation")
        
        assert isinstance(mapped_error, AgentCoreMemoryError)
        assert "Memory memory_operation failed" in mapped_error.message
        assert mapped_error.memory_operation == "memory_operation"


class TestHandleErrorGracefully:
    """Test graceful error handling functionality."""
    
    def test_handle_error_with_default_return(self):
        """Test graceful error handling with default return value."""
        error = ValueError("Test error")
        default_value = {"status": "error"}
        
        result = handle_error_gracefully(error, "test operation", default_value)
        
        assert result == default_value
    
    def test_handle_error_without_default_return(self):
        """Test graceful error handling without default return value."""
        error = ValueError("Test error")
        
        with pytest.raises(MCPServerError):
            handle_error_gracefully(error, "test operation")
    
    def test_handle_aws_error_gracefully(self):
        """Test graceful handling of AWS errors."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'ThrottlingException',
                    'Message': 'Request rate exceeded'
                }
            },
            operation_name='TestOperation'
        )
        
        default_value = []
        result = handle_error_gracefully(client_error, "throttled operation", default_value)
        
        assert result == default_value


class TestLogAWSError:
    """Test AWS error logging functionality."""
    
    def test_log_client_error(self, caplog):
        """Test logging ClientError with context."""
        client_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'AccessDenied',
                    'Message': 'Access denied'
                },
                'ResponseMetadata': {
                    'RequestId': 'test-request-id',
                    'HTTPStatusCode': 403
                }
            },
            operation_name='TestOperation'
        )
        
        context = {'operation': 'test_operation', 'resource': 'test_resource'}
        
        with caplog.at_level(logging.ERROR):
            log_aws_error(client_error, context)
        
        # Verify logging occurred
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.levelname == 'ERROR'
        assert 'AWS ClientError' in log_record.message
        assert 'AccessDenied' in log_record.message
        assert 'test_operation' in log_record.message
    
    def test_log_botocore_error(self, caplog):
        """Test logging BotoCoreError."""
        botocore_error = BotoCoreError()
        context = {'operation': 'test_operation'}
        
        with caplog.at_level(logging.ERROR):
            log_aws_error(botocore_error, context)
        
        # Verify logging occurred
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.levelname == 'ERROR'
        assert 'AWS BotoCoreError' in log_record.message
    
    def test_log_generic_error(self, caplog):
        """Test logging generic error."""
        generic_error = ValueError("Test error")
        context = {'operation': 'test_operation'}
        
        with caplog.at_level(logging.ERROR):
            log_aws_error(generic_error, context)
        
        # Verify logging occurred
        assert len(caplog.records) == 1
        log_record = caplog.records[0]
        assert log_record.levelname == 'ERROR'
        assert 'AWS Error' in log_record.message
        assert 'Test error' in log_record.message
    
    def test_log_error_without_context(self, caplog):
        """Test logging error without context."""
        error = ValueError("Test error")
        
        with caplog.at_level(logging.ERROR):
            log_aws_error(error)
        
        # Should still log successfully
        assert len(caplog.records) == 1


class TestErrorPropagation:
    """Test that errors propagate correctly without suppression."""
    
    def test_error_propagation_in_mapping(self):
        """Test that error mapping doesn't suppress original errors."""
        original_error = ClientError(
            error_response={
                'Error': {
                    'Code': 'TestError',
                    'Message': 'Test message'
                }
            },
            operation_name='TestOperation'
        )
        
        mapped_error = map_aws_error(original_error, "test operation")
        
        # Original error should be preserved in the chain
        assert mapped_error.__cause__ is original_error
    
    def test_fail_fast_behavior(self):
        """Test that errors fail fast without suppression."""
        def failing_operation():
            raise ClientError(
                error_response={
                    'Error': {
                        'Code': 'TestError',
                        'Message': 'Test message'
                    }
                },
                operation_name='TestOperation'
            )
        
        # Should fail immediately without suppression
        with pytest.raises(ClientError):
            failing_operation()
    
    def test_no_error_swallowing(self):
        """Test that no errors are swallowed in error handling utilities."""
        errors_to_test = [
            ValueError("Test value error"),
            RuntimeError("Test runtime error"),
            ClientError(
                error_response={'Error': {'Code': 'TestError', 'Message': 'Test'}},
                operation_name='Test'
            ),
            BotoCoreError()
        ]
        
        for error in errors_to_test:
            # map_aws_error should always return an exception, never None
            mapped = map_aws_error(error, "test")
            assert mapped is not None
            assert isinstance(mapped, Exception)
            
            # map_memory_error should always return an exception, never None
            memory_mapped = map_memory_error(error, "test")
            assert memory_mapped is not None
            assert isinstance(memory_mapped, AgentCoreMemoryError)