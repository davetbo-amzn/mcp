"""Error handling utilities for AgentCore MCP Server.

This module provides custom exception classes and error mapping utilities
for handling AWS service errors and memory-related errors.
"""

import logging
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


class MCPServerError(Exception):
    """Base exception for MCP server errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AWSError(MCPServerError):
    """Exception for AWS service-related errors."""
    
    def __init__(self, message: str, aws_error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, aws_error_code, details)
        self.aws_error_code = aws_error_code


class AgentCoreMemoryError(MCPServerError):
    """Exception for memory-related errors."""
    
    def __init__(self, message: str, memory_operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, memory_operation, details)
        self.memory_operation = memory_operation


class ConfigurationError(MCPServerError):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, config_key, details)
        self.config_key = config_key


def map_aws_error(error: Exception, operation: str = "AWS operation") -> MCPServerError:
    """Map AWS errors to appropriate MCP server exceptions.
    
    Args:
        error: The original AWS error
        operation: Description of the operation that failed
        
    Returns:
        Mapped MCP server exception
    """
    if isinstance(error, ClientError):
        error_code = error.response.get('Error', {}).get('Code', 'Unknown')
        error_message = error.response.get('Error', {}).get('Message', str(error))
        
        # Map specific AWS error codes to more user-friendly messages
        error_mappings = {
            'AccessDenied': f"Access denied for {operation}. Check your AWS permissions.",
            'InvalidParameterValue': f"Invalid parameter provided for {operation}: {error_message}",
            'ResourceNotFound': f"Resource not found for {operation}: {error_message}",
            'ThrottlingException': f"Request throttled for {operation}. Please retry after a delay.",
            'ValidationException': f"Validation error for {operation}: {error_message}",
            'UnauthorizedOperation': f"Unauthorized to perform {operation}. Check your AWS permissions.",
            'NoCredentialsError': "No AWS credentials found. Please configure AWS credentials.",
        }
        
        mapped_message = error_mappings.get(error_code, f"{operation} failed: {error_message}")
        
        details = {
            'aws_error_code': error_code,
            'aws_error_message': error_message,
            'operation': operation,
            'request_id': error.response.get('ResponseMetadata', {}).get('RequestId')
        }
        
        logger.error(f"AWS error in {operation}: {error_code} - {error_message}")
        
        aws_error = AWSError(mapped_message, error_code, details)
        aws_error.__cause__ = error
        return aws_error
    
    elif isinstance(error, BotoCoreError):
        message = f"{operation} failed due to AWS service error: {str(error)}"
        details = {
            'error_type': type(error).__name__,
            'operation': operation
        }
        
        logger.error(f"BotoCore error in {operation}: {str(error)}")
        
        aws_error = AWSError(message, details=details)
        aws_error.__cause__ = error
        return aws_error
    
    else:
        # For non-AWS errors, wrap in generic MCP server error
        message = f"{operation} failed: {str(error)}"
        details = {
            'error_type': type(error).__name__,
            'operation': operation
        }
        
        logger.error(f"Unexpected error in {operation}: {str(error)}")
        
        mcp_error = MCPServerError(message, details=details)
        mcp_error.__cause__ = error
        return mcp_error


def map_memory_error(error: Exception, memory_operation: str) -> AgentCoreMemoryError:
    """Map errors to memory-specific exceptions.
    
    Args:
        error: The original error
        memory_operation: Description of the memory operation that failed
        
    Returns:
        Memory-specific exception
    """
    # First try to map as AWS error if it's AWS-related
    if isinstance(error, (ClientError, BotoCoreError)):
        aws_error = map_aws_error(error, f"memory {memory_operation}")
        return AgentCoreMemoryError(
            aws_error.message,
            memory_operation,
            aws_error.details
        )
    
    # Handle memory-specific error scenarios
    error_message = str(error)
    
    if "strategy" in error_message.lower():
        message = f"Memory strategy error in {memory_operation}: {error_message}"
    elif "session" in error_message.lower():
        message = f"Memory session error in {memory_operation}: {error_message}"
    elif "record" in error_message.lower():
        message = f"Memory record error in {memory_operation}: {error_message}"
    else:
        message = f"Memory {memory_operation} failed: {error_message}"
    
    details = {
        'error_type': type(error).__name__,
        'memory_operation': memory_operation
    }
    
    logger.error(f"Memory error in {memory_operation}: {error_message}")
    
    return AgentCoreMemoryError(message, memory_operation, details)


def handle_error_gracefully(error: Exception, operation: str, default_return=None):
    """Handle errors gracefully with logging and optional default return.
    
    Args:
        error: The error that occurred
        operation: Description of the operation
        default_return: Default value to return instead of raising
        
    Returns:
        Default return value if provided, otherwise raises mapped exception
    """
    mapped_error = map_aws_error(error, operation)
    
    if default_return is not None:
        logger.warning(f"Handling error gracefully for {operation}: {mapped_error.message}")
        return default_return
    else:
        raise mapped_error


def log_aws_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log AWS errors with context information.
    
    This is a simple logging utility for AWS errors. For more comprehensive
    error handling, use map_aws_error() instead.
    
    Args:
        error: The AWS error to log
        context: Additional context information
    """
    context = context or {}
    
    if isinstance(error, ClientError):
        error_info = {
            'error_code': error.response.get('Error', {}).get('Code'),
            'error_message': error.response.get('Error', {}).get('Message'),
            'request_id': error.response.get('ResponseMetadata', {}).get('RequestId'),
            'http_status': error.response.get('ResponseMetadata', {}).get('HTTPStatusCode'),
            **context
        }
        logger.error(f"AWS ClientError: {error_info}")
    elif isinstance(error, BotoCoreError):
        logger.error(f"AWS BotoCoreError: {str(error)}, Context: {context}")
    else:
        logger.error(f"AWS Error: {str(error)}, Context: {context}")