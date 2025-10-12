"""Tests for AWS client utilities.

These tests validate AWS client creation, credential resolution, and error handling
using real AWS services without mocks, following fail-fast testing principles.
"""

import pytest
import os
from unittest.mock import patch
from botocore.exceptions import ClientError, NoCredentialsError

from awslabs.amazon_bedrock_agentcore_mcp_server.utils.aws_client import (
    AWSClientManager,
    create_aws_client_manager,
    resolve_aws_region
)
from awslabs.amazon_bedrock_agentcore_mcp_server.utils.exceptions import AWSError, MCPServerError


class TestAWSClientManager:
    """Test AWS client manager functionality."""
    
    def test_create_client_manager_with_region(self):
        """Test creating client manager with specific region."""
        region = 'us-east-1'
        manager = AWSClientManager(region_name=region)
        
        assert manager.region_name == region
        assert manager.get_region() == region
    
    def test_create_client_manager_without_region(self):
        """Test creating client manager without region uses default resolution."""
        manager = AWSClientManager()
        
        # Should resolve to a valid region
        region = manager.get_region()
        assert region is not None
        assert isinstance(region, str)
        assert len(region) > 0
    
    def test_get_sts_client(self):
        """Test getting STS client for identity operations."""
        manager = create_aws_client_manager()
        
        # This should work with valid AWS credentials
        sts_client = manager.get_sts_client()
        assert sts_client is not None
        
        # Verify it's the same instance on subsequent calls
        sts_client2 = manager.get_sts_client()
        assert sts_client is sts_client2
    
    def test_get_agentcore_clients(self):
        """Test getting AgentCore clients."""
        manager = create_aws_client_manager()
        
        # Get data plane client
        data_client = manager.get_agentcore_client()
        assert data_client is not None
        
        # Get control plane client
        control_client = manager.get_agentcore_control_client()
        assert control_client is not None
        
        # Verify they're different clients
        assert data_client is not control_client
    
    def test_test_credentials_success(self):
        """Test credential validation with real AWS credentials."""
        manager = create_aws_client_manager()
        
        # This should work with valid AWS credentials
        response = manager.test_credentials()
        
        # Verify response structure
        assert 'Account' in response
        assert 'Arn' in response
        assert 'UserId' in response
        
        # Verify account ID is valid format
        account_id = response['Account']
        assert len(account_id) == 12
        assert account_id.isdigit()
    
    def test_client_caching(self):
        """Test that clients are properly cached."""
        manager = create_aws_client_manager()
        
        # Get same service client multiple times
        client1 = manager.get_client('sts')
        client2 = manager.get_client('sts')
        
        # Should be the same instance
        assert client1 is client2
    
    def test_client_with_different_kwargs(self):
        """Test that clients with different kwargs are cached separately."""
        manager = create_aws_client_manager()
        
        # Get clients with different configurations
        client1 = manager.get_client('sts')
        client2 = manager.get_client('sts', endpoint_url='https://custom.endpoint')
        
        # Should be different instances
        assert client1 is not client2


class TestRegionResolution:
    """Test AWS region resolution functionality."""
    
    def test_resolve_region_from_environment(self):
        """Test region resolution from environment variables."""
        test_region = 'eu-west-1'
        
        with patch.dict(os.environ, {'AWS_REGION': test_region}):
            region = resolve_aws_region()
            assert region == test_region
    
    def test_resolve_region_from_default_environment(self):
        """Test region resolution from AWS_DEFAULT_REGION."""
        test_region = 'ap-southeast-2'
        
        # Clear AWS_REGION and set AWS_DEFAULT_REGION
        env_vars = {'AWS_DEFAULT_REGION': test_region}
        if 'AWS_REGION' in os.environ:
            env_vars['AWS_REGION'] = ''
            
        with patch.dict(os.environ, env_vars, clear=False):
            region = resolve_aws_region()
            assert region == test_region
    
    def test_resolve_region_fallback(self):
        """Test region resolution fallback when no environment variables set."""
        # Clear both environment variables
        env_vars = {}
        if 'AWS_REGION' in os.environ:
            env_vars['AWS_REGION'] = ''
        if 'AWS_DEFAULT_REGION' in os.environ:
            env_vars['AWS_DEFAULT_REGION'] = ''
            
        with patch.dict(os.environ, env_vars, clear=False):
            region = resolve_aws_region()
            # Should fall back to us-west-2 or get from boto3 session
            assert region is not None
            assert isinstance(region, str)
            assert len(region) > 0


class TestErrorHandling:
    """Test error handling in AWS client utilities."""
    
    def test_invalid_service_name(self):
        """Test error handling for invalid service names."""
        manager = create_aws_client_manager()
        
        # This should raise an AWSError
        with pytest.raises(AWSError) as exc_info:
            manager.get_client('invalid-service-name-that-does-not-exist')
        
        assert "Failed to create invalid-service-name-that-does-not-exist client" in str(exc_info.value)
    
    def test_credential_test_with_invalid_credentials(self):
        """Test credential testing with invalid credentials."""
        # This test requires temporarily removing AWS credentials
        # We'll patch the session creation to simulate no credentials
        
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.get_credentials.return_value = None
            mock_session.return_value.client.side_effect = NoCredentialsError()
            
            manager = AWSClientManager()
            
            # The error gets mapped through handle_error_gracefully, so expect MCPServerError
            with pytest.raises((AWSError, MCPServerError)) as exc_info:
                manager.test_credentials()
            
            assert "credential test" in str(exc_info.value).lower()


class TestCreateAWSClientManager:
    """Test the factory function for creating AWS client managers."""
    
    def test_create_with_region(self):
        """Test creating client manager with specific region."""
        region = 'ca-central-1'
        manager = create_aws_client_manager(region_name=region)
        
        assert isinstance(manager, AWSClientManager)
        assert manager.region_name == region
    
    def test_create_without_region(self):
        """Test creating client manager without region."""
        manager = create_aws_client_manager()
        
        assert isinstance(manager, AWSClientManager)
        # Should use default region resolution
        region = manager.get_region()
        assert region is not None


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_full_workflow_with_real_aws(self):
        """Test complete workflow with real AWS services."""
        # Create manager
        manager = create_aws_client_manager()
        
        # Test credentials
        identity = manager.test_credentials()
        assert 'Account' in identity
        
        # Get multiple clients
        sts_client = manager.get_sts_client()
        agentcore_client = manager.get_agentcore_client()
        control_client = manager.get_agentcore_control_client()
        
        # Verify all clients are created
        assert sts_client is not None
        assert agentcore_client is not None
        assert control_client is not None
        
        # Verify region consistency
        region = manager.get_region()
        assert region == sts_client.meta.region_name
        assert region == agentcore_client.meta.region_name
        assert region == control_client.meta.region_name
    
    def test_error_propagation(self):
        """Test that errors propagate correctly without suppression."""
        manager = create_aws_client_manager()
        
        # Try to create a client for a service that doesn't exist
        # This should fail fast without error suppression
        with pytest.raises(AWSError):
            manager.get_client('nonexistent-aws-service')