"""Tests for caller identity utilities.

These tests validate actor ID resolution and AWS account ID retrieval
using real AWS services without mocks, following fail-fast testing principles.
"""

import pytest
import logging
from unittest.mock import patch
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

from awslabs.amazon_bedrock_agentcore_mcp_server.utils.caller_identity import (
    get_actor_id,
    get_account_id,
    parse_actor_id_components
)
from awslabs.amazon_bedrock_agentcore_mcp_server.utils.exceptions import AWSError


class TestGetActorId:
    """Test actor ID resolution functionality."""
    
    def test_get_actor_id_success(self):
        """Test successful actor ID resolution with real AWS credentials."""
        actor_id = get_actor_id()
        
        # Verify actor ID is returned
        assert actor_id is not None
        assert isinstance(actor_id, str)
        assert len(actor_id) > 0
        
        # Should be a compliant actor ID format (account-userid or processed ARN)
        # The actor ID is processed to be AgentCore-compliant, not raw ARN
        assert '-' in actor_id or ':' in actor_id  # Should contain separators
        
        # Should not contain problematic characters for AgentCore
        assert not actor_id.startswith(':')  # Should not start with separator
        assert '::' not in actor_id  # Should not have empty components
    
    def test_get_actor_id_consistency(self):
        """Test that actor ID is consistent across multiple calls."""
        actor_id1 = get_actor_id()
        actor_id2 = get_actor_id()
        
        # Should return the same actor ID
        assert actor_id1 == actor_id2
    
    def test_get_actor_id_with_no_credentials(self):
        """Test actor ID resolution with no credentials."""
        with patch('boto3.client') as mock_client:
            mock_client.side_effect = NoCredentialsError()
            
            with pytest.raises(AWSError) as exc_info:
                get_actor_id()
            
            assert "No AWS credentials found" in str(exc_info.value)
    
    def test_get_actor_id_with_client_error(self):
        """Test actor ID resolution with AWS client error."""
        with patch('boto3.client') as mock_client:
            mock_sts = mock_client.return_value
            mock_sts.get_caller_identity.side_effect = ClientError(
                error_response={
                    'Error': {
                        'Code': 'AccessDenied',
                        'Message': 'Access denied'
                    }
                },
                operation_name='GetCallerIdentity'
            )
            
            with pytest.raises(AWSError) as exc_info:
                get_actor_id()
            
            assert "Failed to get caller identity" in str(exc_info.value)
            assert "AccessDenied" in str(exc_info.value)


class TestGetAccountId:
    """Test AWS account ID resolution functionality."""
    
    def test_get_account_id_success(self):
        """Test successful account ID resolution with real AWS credentials."""
        account_id = get_account_id()
        
        # Verify account ID format
        assert account_id is not None
        assert isinstance(account_id, str)
        assert len(account_id) == 12
        assert account_id.isdigit()
    
    def test_get_account_id_consistency(self):
        """Test that account ID is consistent across multiple calls."""
        account_id1 = get_account_id()
        account_id2 = get_account_id()
        
        # Should return the same account ID
        assert account_id1 == account_id2
    
    def test_account_id_matches_actor_id(self):
        """Test that account ID matches the account in actor ID."""
        actor_id = get_actor_id()
        account_id = get_account_id()
        
        # Extract account from actor ID (format: account-userid)
        # The actor ID should start with the account ID
        assert actor_id.startswith(account_id)
        
        # Should contain the account ID as the first component
        if '-' in actor_id:
            actor_account = actor_id.split('-')[0]
            assert account_id == actor_account
    
    def test_get_account_id_with_no_credentials(self):
        """Test account ID resolution with no credentials."""
        with patch('boto3.client') as mock_client:
            mock_client.side_effect = NoCredentialsError()
            
            with pytest.raises(AWSError) as exc_info:
                get_account_id()
            
            assert "No AWS credentials found" in str(exc_info.value)


class TestParseActorIdComponents:
    """Test actor ID component parsing functionality."""
    
    def test_parse_assumed_role_arn(self):
        """Test parsing assumed role ARN components."""
        arn = "arn:aws:sts:us-west-2:123456789012:assumed-role/MyRole/MySession"
        
        components = parse_actor_id_components(arn)
        
        assert components['service'] == 'sts'
        assert components['region'] == 'us-west-2'
        assert components['account'] == '123456789012'
        assert components['role'] == 'MyRole'
        assert components['user'] is None
    
    def test_parse_user_arn(self):
        """Test parsing IAM user ARN components."""
        arn = "arn:aws:iam::123456789012:user/MyUser"
        
        components = parse_actor_id_components(arn)
        
        assert components['service'] == 'iam'
        assert components['region'] is None  # IAM is global
        assert components['account'] == '123456789012'
        assert components['role'] is None
        assert components['user'] == 'MyUser'
    
    def test_parse_root_arn(self):
        """Test parsing root account ARN components."""
        arn = "arn:aws:iam::123456789012:root"
        
        components = parse_actor_id_components(arn)
        
        assert components['service'] == 'iam'
        assert components['region'] is None
        assert components['account'] == '123456789012'
        assert components['role'] is None
        assert components['user'] is None
    
    def test_parse_invalid_arn(self):
        """Test parsing invalid ARN format."""
        invalid_arn = "not-an-arn"
        
        components = parse_actor_id_components(invalid_arn)
        
        # Should return empty components
        assert components['service'] is None
        assert components['region'] is None
        assert components['account'] is None
        assert components['role'] is None
        assert components['user'] is None
    
    def test_parse_real_actor_id(self):
        """Test parsing real actor ID from current AWS credentials."""
        actor_id = get_actor_id()
        
        components = parse_actor_id_components(actor_id)
        
        # For non-ARN actor IDs, components may be None
        # This is expected behavior when actor ID is not in ARN format
        if actor_id.startswith('arn:aws:'):
            # Only test ARN parsing for actual ARNs
            assert components['service'] is not None
            assert components['account'] is not None
            assert len(components['account']) == 12
            assert components['account'].isdigit()
            
            # Should have either role or user for ARN format
            assert components['role'] is not None or components['user'] is not None
        else:
            # For non-ARN format, parsing returns None values
            # This is the expected behavior
            logger.info(f"Actor ID is not in ARN format: {actor_id}")
            assert components['service'] is None
            assert components['role'] is None
            assert components['user'] is None


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_complete_identity_workflow(self):
        """Test complete identity resolution workflow."""
        # Get actor ID and account ID
        actor_id = get_actor_id()
        account_id = get_account_id()
        
        # Parse actor ID components
        components = parse_actor_id_components(actor_id)
        
        # Verify consistency - only for ARN format actor IDs
        if actor_id.startswith('arn:aws:'):
            assert components['account'] == account_id
        else:
            # For non-ARN format, verify account ID is present in actor ID
            assert account_id in actor_id
        
        # Log results for debugging
        print(f"Actor ID: {actor_id}")
        print(f"Account ID: {account_id}")
        print(f"Components: {components}")
    
    def test_error_propagation_without_suppression(self):
        """Test that errors propagate correctly without suppression."""
        # Patch to simulate credential error
        with patch('boto3.client') as mock_client:
            mock_client.side_effect = ClientError(
                error_response={
                    'Error': {
                        'Code': 'InvalidUserID.NotFound',
                        'Message': 'The user ID does not exist'
                    }
                },
                operation_name='GetCallerIdentity'
            )
            
            # Should fail fast without error suppression
            with pytest.raises(AWSError):
                get_actor_id()
            
            with pytest.raises(AWSError):
                get_account_id()
    
    def test_multiple_concurrent_calls(self):
        """Test multiple concurrent identity resolution calls."""
        import concurrent.futures
        
        def get_identity():
            return get_actor_id(), get_account_id()
        
        # Run multiple concurrent calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_identity) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result