"""AWS client utilities for AgentCore MCP Server.

This module provides utilities for creating and managing AWS service clients
with proper credential resolution and error handling.
"""

import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Dict, Any

from .exceptions import AWSError, handle_error_gracefully

logger = logging.getLogger(__name__)


class AWSClientManager:
    """Manages AWS service clients with credential resolution and error handling."""
    
    def __init__(self, region_name: Optional[str] = None):
        """Initialize AWS client manager.
        
        Args:
            region_name: AWS region name. If None, uses default region resolution.
        """
        self.region_name = region_name or resolve_aws_region()
        self._session = None
        self._clients = {}
        
    def _get_session(self) -> boto3.Session:
        """Get or create a boto3 session with credential validation.
        
        Returns:
            Configured boto3 session
            
        Raises:
            AWSError: If credentials cannot be resolved
        """
        if self._session is None:
            try:
                # Create session with region
                self._session = boto3.Session(region_name=self.region_name)
                
                # Validate credentials early
                credentials = self._session.get_credentials()
                if credentials is None:
                    logger.warning(
                        "No AWS credentials found during initialization. "
                        "Credentials may be available through IAM roles, instance metadata, "
                        "container credentials, or other AWS credential providers."
                    )
                else:
                    logger.info("AWS credentials resolved successfully")
                    
            except NoCredentialsError as e:
                error_msg = (
                    "No AWS credentials found. Please configure AWS credentials using "
                    "one of the following methods: AWS CLI, environment variables, "
                    "IAM roles, or instance metadata."
                )
                logger.error(error_msg)
                raise AWSError(error_msg) from e
                
        return self._session
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """Get or create an AWS service client.
        
        Args:
            service_name: AWS service name (e.g., 'bedrock-agentcore', 'sts')
            **kwargs: Additional arguments to pass to client creation
            
        Returns:
            AWS service client
            
        Raises:
            AWSError: If client creation fails
        """
        client_key = f"{service_name}_{hash(frozenset(kwargs.items()))}"
        
        if client_key not in self._clients:
            try:
                session = self._get_session()
                
                # Merge region with any provided kwargs
                client_kwargs = {'region_name': self.region_name, **kwargs}
                
                self._clients[client_key] = session.client(service_name, **client_kwargs)
                logger.debug(f"Created {service_name} client for region: {self.region_name}")
                
            except Exception as e:
                error_msg = f"Failed to create {service_name} client"
                logger.error(f"{error_msg}: {str(e)}")
                raise AWSError(error_msg) from e
                
        return self._clients[client_key]
    
    def get_agentcore_client(self):
        """Get AgentCore data plane client.
        
        Returns:
            AgentCore data plane client
        """
        return self.get_client('bedrock-agentcore')
    
    def get_agentcore_control_client(self):
        """Get AgentCore control plane client.
        
        Returns:
            AgentCore control plane client
        """
        return self.get_client('bedrock-agentcore-control')
    
    def get_sts_client(self):
        """Get STS client for identity operations.
        
        Returns:
            STS client
        """
        return self.get_client('sts')
    
    def test_credentials(self) -> Dict[str, Any]:
        """Test AWS credentials by calling STS get_caller_identity.
        
        Returns:
            Caller identity information
            
        Raises:
            AWSError: If credential test fails
        """
        try:
            sts_client = self.get_sts_client()
            response = sts_client.get_caller_identity()
            
            logger.info(f"Credential test successful. Account: {response.get('Account')}")
            return response
            
        except Exception as e:
            raise handle_error_gracefully(e, "credential test")
    
    def get_region(self) -> str:
        """Get the configured AWS region.
        
        Returns:
            AWS region name
        """
        return self.region_name


def create_aws_client_manager(region_name: Optional[str] = None) -> AWSClientManager:
    """Create an AWS client manager instance.
    
    Args:
        region_name: AWS region name. If None, uses default region resolution.
        
    Returns:
        Configured AWS client manager
    """
    return AWSClientManager(region_name=region_name)


def resolve_aws_region() -> str:
    """Resolve AWS region from various sources.
    
    Checks environment variables and AWS configuration in order:
    1. AWS_REGION
    2. AWS_DEFAULT_REGION
    3. boto3 session default
    4. Fallback to us-west-2
    
    Returns:
        AWS region name
    """
    import os
    
    # Check environment variables first
    region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
    if region:
        logger.debug(f"Using region from environment: {region}")
        return region
    
    # Try boto3 session default
    try:
        session = boto3.Session()
        region = session.region_name
        if region:
            logger.debug(f"Using region from boto3 session: {region}")
            return region
    except Exception as e:
        logger.debug(f"Could not get region from boto3 session: {e}")
    
    # Fallback
    region = 'us-west-2'
    logger.warning(f"No region found, using fallback: {region}")
    return region