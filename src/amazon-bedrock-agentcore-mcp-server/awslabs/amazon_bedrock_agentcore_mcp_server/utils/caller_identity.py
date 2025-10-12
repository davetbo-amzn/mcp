"""Actor ID resolution utilities.

This module provides utilities for resolving actor IDs from AWS credentials
and caller identity information.
"""

import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional

from .exceptions import AWSError

logger = logging.getLogger(__name__)


def get_actor_id() -> str:
    """Get the current actor ID from AWS caller identity.
    
    Resolves the actor ID by calling AWS STS get_caller_identity and
    extracting the appropriate identifier from the response.
    
    The actor ID format follows the pattern used in the reference implementation:
    - Get the full ARN from STS get_caller_identity
    - Replace '::' with ':'
    - Replace ':' with '-'
    - Replace '/' with '-'
    - Remove the 'arn-aws-sts-' prefix
    
    Example: arn:aws:sts::165361166149:assumed-role/Admin/davetbo-Isengard
    becomes: 165361166149-assumed-role-Admin-davetbo-Isengard
    
    Returns:
        Actor ID string suitable for use with AgentCore memory services
        
    Raises:
        AWSError: If unable to resolve caller identity or extract actor ID
    """
    try:
        # Create STS client to get caller identity
        sts_client = boto3.client('sts')
        
        # Get caller identity
        response = sts_client.get_caller_identity()
        
        # Extract ARN from the response
        arn = response.get('Arn')
        if not arn:
            raise AWSError("No ARN found in caller identity response")
        
        logger.debug(f"Resolved caller identity ARN: {arn}")
        
        # Create actor ID following the reference implementation format
        # 1. Replace :: with :
        # 2. Replace : with -
        # 3. Replace / with -
        # 4. Remove arn-aws-sts- prefix
        actor_id = arn.replace('::', ':').replace(':', '-').replace('/', '-').replace('arn-aws-sts-', '')

        logger.debug(f"Created compliant actor ID: {actor_id}")
        return actor_id
        
    except NoCredentialsError as e:
        error_msg = "No AWS credentials found. Please configure AWS credentials."
        logger.error(error_msg)
        raise AWSError(error_msg) from e
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = f"Failed to get caller identity: {error_code} - {str(e)}"
        logger.error(error_msg)
        raise AWSError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error resolving actor ID: {str(e)}"
        logger.error(error_msg)
        raise AWSError(error_msg) from e


def get_account_id() -> str:
    """Get the current AWS account ID.
    
    Returns:
        AWS account ID string
        
    Raises:
        AWSError: If unable to resolve account ID
    """
    try:
        sts_client = boto3.client('sts')
        response = sts_client.get_caller_identity()
        
        account_id = response.get('Account')
        if not account_id:
            raise AWSError("No Account ID found in caller identity response")
            
        logger.debug(f"Resolved AWS account ID: {account_id}")
        return account_id
        
    except NoCredentialsError as e:
        error_msg = "No AWS credentials found. Please configure AWS credentials."
        logger.error(error_msg)
        raise AWSError(error_msg) from e
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = f"Failed to get account ID: {error_code} - {str(e)}"
        logger.error(error_msg)
        raise AWSError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Unexpected error resolving account ID: {str(e)}"
        logger.error(error_msg)
        raise AWSError(error_msg) from e


def parse_actor_id_components(actor_id: str) -> dict[str, Optional[str]]:
    """Parse components from an actor ID ARN.
    
    Args:
        actor_id: Actor ID (typically an AWS ARN)
        
    Returns:
        Dictionary with parsed components (account, role, user, etc.)
    """
    components = {
        'account': None,
        'role': None,
        'user': None,
        'service': None,
        'region': None
    }
    
    if not actor_id.startswith('arn:aws:'):
        logger.warning(f"Actor ID does not appear to be an AWS ARN: {actor_id}")
        return components
    
    try:
        # Parse ARN format: arn:aws:service:region:account:resource
        parts = actor_id.split(':')
        if len(parts) >= 5:
            components['service'] = parts[2]
            components['region'] = parts[3] if parts[3] else None
            components['account'] = parts[4]
            
            # Extract role or user from resource part
            if len(parts) >= 6:
                resource = ':'.join(parts[5:])  # Join remaining parts
                if 'assumed-role/' in resource:
                    # Extract role name from assumed-role/RoleName/SessionName
                    role_parts = resource.split('/')
                    if len(role_parts) >= 2:
                        components['role'] = role_parts[1]
                elif 'user/' in resource:
                    # Extract user name from user/UserName
                    user_parts = resource.split('/')
                    if len(user_parts) >= 2:
                        components['user'] = user_parts[1]
                        
    except Exception as e:
        logger.warning(f"Failed to parse actor ID components: {e}")
    
    return components