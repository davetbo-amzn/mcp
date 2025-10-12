"""AgentCore memory client for MCP server.

This module provides the main client interface for interacting with AgentCore memory services.
It handles memory event operations, memory record operations, and integrates with AWS services.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from botocore.exceptions import ClientError, BotoCoreError

from ..utils.caller_identity import get_actor_id
from ..utils.exceptions import AgentCoreMemoryError, AWSError, handle_error_gracefully, log_aws_error
from ..utils.aws_client import create_aws_client_manager
from .config import MemoryConfig
from .memory_strategy_config import MemoryStrategyConfig
from .types import MemoryEvent, ConversationMessage, MemoryRecord

logger = logging.getLogger(__name__)


class AgentCoreMemoryClient:
    """Client for interacting with AgentCore memory services.
    
    This client provides methods for managing memory events, retrieving memory records,
    and handling memory strategy operations.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory client.
        
        Args:
            config: Memory configuration. If None, loads from environment.
        """
        self.config = config or MemoryConfig.from_environment()
        self.strategy_config = MemoryStrategyConfig()
        self._actor_id: Optional[str] = None
        self._memory_provider: Optional[Dict[str, Any]] = None
        
        # Initialize AWS clients
        self.aws_manager = create_aws_client_manager(self.config.aws_region)
        
        logger.info(f"Initialized AgentCore memory client for region: {self.config.aws_region}")
    
    @property
    def actor_id(self) -> str:
        """Get the current actor ID, resolving it if necessary."""
        if self._actor_id is None:
            self._actor_id = get_actor_id()
        return self._actor_id
    
    @property
    def memory_id(self) -> str:
        """Get the memory ID from configuration.
        
        Returns:
            Memory ID
            
        Raises:
            AgentCoreMemoryError: If memory ID is not configured
        """
        if not self.config.memory_id:
            raise AgentCoreMemoryError(
                "Memory ID not configured. Please set AGENTCORE_MCP_MEMORY_ID environment variable."
            )
        return self.config.memory_id
    
    def _create_session_id(self) -> str:
        """Create a unique session ID.
        
        Returns:
            Formatted session ID with timestamp that matches AWS validation pattern
        """
        # Create timestamp that matches pattern [a-zA-Z0-9][a-zA-Z0-9-_]*
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        return f"session-{timestamp}"
    
    def create_memory_event(
        self,
        messages: List[ConversationMessage],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a memory event with conversation messages.
        
        Args:
            messages: List of conversation messages
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing the created event details
            
        Raises:
            AgentCoreMemoryError: If memory event creation fails
        """
        if not session_id:
            session_id = self._create_session_id()
        
        try:
            # Format messages for AgentCore API
            formatted_messages = []
            for message in messages:
                formatted_messages.append({
                    'conversational': {
                        'content': {
                            'text': message.content
                        },
                        'role': message.role
                    }
                })
            
            logger.info(f"Creating event for session_id {session_id}")
            
            client = self.aws_manager.get_agentcore_client()
            response = client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=session_id,
                eventTimestamp=datetime.now(timezone.utc),
                payload=formatted_messages
            )
            
            logger.info(f"Created event for session {session_id}, actor {self.actor_id}")
            return response
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'create_event',
                'memory_id': self.memory_id,
                'actor_id': self.actor_id,
                'session_id': session_id
            })
            raise handle_error_gracefully(e, "create memory event")
    
    def list_memory_events(
        self,
        session_id: str,
        max_results: int = 15
    ) -> List[MemoryEvent]:
        """List memory events for a session.
        
        Args:
            session_id: Session identifier
            max_results: Maximum number of results to return
            
        Returns:
            List of memory events
            
        Raises:
            AgentCoreMemoryError: If listing memory events fails
        """
        try:
            client = self.aws_manager.get_agentcore_client()
            
            response = client.list_events(
                memoryId=self.memory_id,
                sessionId=session_id,
                maxResults=max_results,
                actorId=self.actor_id,
                includePayloads=True
            )
            
            # Convert AWS response to MemoryEvent objects
            events = []
            for event_data in response.get('events', []):
                # Parse messages from payload
                messages = []
                for payload_item in event_data.get('payload', []):
                    if 'conversational' in payload_item:
                        conv = payload_item['conversational']
                        content = conv.get('content', {}).get('text', '')
                        role = conv.get('role', 'USER')
                        messages.append(ConversationMessage(role=role, content=content))
                
                event = MemoryEvent(
                    event_id=event_data.get('eventId', ''),
                    session_id=event_data.get('sessionId', session_id),
                    actor_id=event_data.get('actorId', self.actor_id),
                    timestamp=event_data.get('eventTimestamp', datetime.now(timezone.utc)),
                    messages=messages
                )
                events.append(event)
            
            logger.info(f"Listed {len(events)} events for session {session_id}")
            return events
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'list_events',
                'memory_id': self.memory_id,
                'actor_id': self.actor_id,
                'session_id': session_id
            })
            raise handle_error_gracefully(e, "list memory events")
    
    
    def retrieve_memories(
        self,
        query: str,
        memory_strategy_type: str,
        session_id: Optional[str] = None,
        max_results: int = 15
    ) -> List[MemoryRecord]:
        """Retrieve memory records using semantic search.
        
        Args:
            query: Query string for memory retrieval
            memory_strategy_type: The memory strategy type to use
            session_id: Optional session identifier
            max_results: Maximum number of results to return
            
        Returns:
            List of memory records
            
        Raises:
            AgentCoreMemoryError: If memory retrieval fails
        """
        try:
            # Validate and get strategy ID
            if not self.strategy_config.is_valid_strategy_type(memory_strategy_type):
                raise AgentCoreMemoryError(f"Invalid memory strategy type: {memory_strategy_type}")
            
            strategy_id = self.strategy_config.get_strategy_id(memory_strategy_type)
            if not strategy_id:
                raise AgentCoreMemoryError(f"Memory strategy '{memory_strategy_type}' not configured")
            
            # Get memory provider details and resolve namespace
            namespace = self._resolve_namespace(strategy_id, session_id)
            
            client = self.aws_manager.get_agentcore_client()
            
            response = client.retrieve_memory_records(
                memoryId=self.memory_id,
                namespace=namespace,
                searchCriteria={
                    'searchQuery': query
                },
                maxResults=max_results
            )
            
            # Convert AWS response to MemoryRecord objects
            records = []
            for record_data in response.get('memoryRecords', []):
                record = MemoryRecord(
                    memory_record_id=record_data.get('memoryRecordId', ''),
                    content=record_data.get('content', ''),
                    timestamp=record_data.get('timestamp', datetime.now(timezone.utc)),
                    relevance_score=record_data.get('relevanceScore', 0.0),
                    namespace=namespace
                )
                records.append(record)
            
            logger.info(f"Retrieved {len(records)} memory records for query: {query[:50]}...")
            return records
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'retrieve_memory_records',
                'memory_id': self.memory_id,
                'query': query[:100]
            })
            raise handle_error_gracefully(e, "retrieve memories")
    
    def list_actors(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """List actors (participants) in memory conversations.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            List of actor information
            
        Raises:
            AgentCoreMemoryError: If listing actors fails
        """
        try:
            client = self.aws_manager.get_agentcore_client()
            
            logger.info(f'Listing actors for memory id {self.memory_id}')
            response = client.list_actors(
                memoryId=self.memory_id,
                maxResults=max_results
            )
            
            actors = response.get('actorSummaries', [])
            logger.info(f"list_actors got {len(actors)} actors")
            return actors
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'list_actors',
                'memory_id': self.memory_id
            })
            raise handle_error_gracefully(e, "list actors")

    def list_sessions(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """List conversation sessions for the current AWS caller identity.

        Args:
            max_results: Maximum number of results (default: 10)

        Returns:
            List of session summaries

        Raises:
            AgentCoreMemoryError: If listing sessions fails

        Note: Sessions are automatically filtered by the AWS STS caller identity ARN.
        """
        try:
            client = self.aws_manager.get_agentcore_client()

            logger.info(f'Listing sessions for memory id {self.memory_id}')
            response = client.list_sessions(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                maxResults=max_results
            )['sessionSummaries']

            logger.info(f"list_sessions got {len(response)} sessions")
            return response

        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'list_sessions',
                'memory_id': self.memory_id,
                'actor_id': self.actor_id
            })
            raise handle_error_gracefully(e, "list sessions")
    
    def get_memory_provider_details(self) -> Dict[str, Any]:
        """Get memory provider details from control plane.
        
        Returns:
            Memory provider details
            
        Raises:
            AgentCoreMemoryError: If retrieval fails
        """
        if self._memory_provider is None:
            try:
                control_client = self.aws_manager.get_agentcore_control_client()
                response = control_client.get_memory(memoryId=self.memory_id)
                self._memory_provider = response['memory']
                logger.info(f"Retrieved memory provider details for {self.memory_id}")
            except (ClientError, BotoCoreError) as e:
                log_aws_error(e, {
                    'operation': 'get_memory',
                    'memory_id': self.memory_id
                })
                raise handle_error_gracefully(e, "get memory provider details")
        
        return self._memory_provider
    
    def _resolve_namespace(self, strategy_id: str, session_id: Optional[str] = None) -> str:
        """Resolve namespace for a memory strategy.
        
        Args:
            strategy_id: Memory strategy ID
            session_id: Optional session ID for namespace resolution
            
        Returns:
            Resolved namespace string
            
        Raises:
            AgentCoreMemoryError: If namespace resolution fails
        """
        try:
            memory_provider = self.get_memory_provider_details()
            
            namespace = ''
            for strategy in memory_provider.get('strategies', []):
                if strategy.get('strategyId') == strategy_id:
                    namespaces = strategy.get('namespaces', [])
                    if namespaces:
                        namespace = namespaces[0]
                        # Replace template variables
                        namespace = namespace.replace('{actorId}', self.actor_id)
                        namespace = namespace.replace('{memoryStrategyId}', strategy_id)
                        
                        # Handle session ID replacement
                        if '{sessionId}' in namespace:
                            if session_id is not None:
                                namespace = namespace.replace('{sessionId}', session_id)
                            else:
                                # Remove session ID part if not provided
                                namespace = namespace.replace('/{sessionId}', '')
                        break
            
            if not namespace:
                raise AgentCoreMemoryError(f"Could not resolve namespace for strategy: {strategy_id}")
            
            logger.info(f'Resolved namespace: {namespace}')
            return namespace
            
        except Exception as e:
            if isinstance(e, AgentCoreMemoryError):
                raise
            raise AgentCoreMemoryError(f"Failed to resolve namespace: {str(e)}")
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of available memory strategies from the provider.
        
        Returns:
            List of strategy information dictionaries
            
        Raises:
            AgentCoreMemoryError: If unable to retrieve strategies
        """
        try:
            memory_provider = self.get_memory_provider_details()
            strategies = memory_provider.get('strategies', [])
            
            logger.info(f"Retrieved {len(strategies)} available memory strategies")
            return strategies
            
        except Exception as e:
            if isinstance(e, AgentCoreMemoryError):
                raise
            raise AgentCoreMemoryError(f"Failed to get available strategies: {str(e)}")
    
    def get_strategy_by_type(self, strategy_type: str) -> Optional[Dict[str, Any]]:
        """Get strategy details by strategy type.
        
        Args:
            strategy_type: Memory strategy type (e.g., 'semantic', 'user_preferences')
            
        Returns:
            Strategy details dictionary or None if not found
        """
        try:
            strategy_id = self.strategy_config.get_strategy_id(strategy_type)
            if not strategy_id:
                return None
            
            strategies = self.get_available_strategies()
            for strategy in strategies:
                if strategy.get('strategyId') == strategy_id:
                    return strategy
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get strategy by type {strategy_type}: {e}")
            return None
    
    def validate_memory_configuration(self) -> Dict[str, Any]:
        """Validate the current memory configuration.
        
        Returns:
            Dictionary with validation results and configuration status
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'memory_id_configured': bool(self.config.memory_id),
            'strategies_configured': self.config.get_configured_strategies(),
            'available_strategies': []
        }
        
        # Check memory ID
        if not self.config.memory_id:
            validation_result['valid'] = False
            validation_result['errors'].append("Memory ID not configured (AGENTCORE_MCP_MEMORY_ID)")
        
        # Check available strategies if memory ID is configured
        if self.config.memory_id:
            try:
                available_strategies = self.get_available_strategies()
                validation_result['available_strategies'] = [
                    s.get('strategyId', 'unknown') for s in available_strategies
                ]
                
                # Check if configured strategies are available
                configured = self.config.get_configured_strategies()
                available_ids = [s.get('strategyId', '') for s in available_strategies]
                
                for strategy_type in configured:
                    strategy_id = self.strategy_config.get_strategy_id(strategy_type)
                    if strategy_id not in available_ids:
                        validation_result['warnings'].append(
                            f"Configured strategy '{strategy_type}' (ID: {strategy_id}) not found in memory provider"
                        )
                        
            except Exception as e:
                validation_result['warnings'].append(f"Could not validate strategies: {str(e)}")
        
        logger.info(f"Memory configuration validation: {'valid' if validation_result['valid'] else 'invalid'}")
        return validation_result
    
    def delete_event(self, event_id: str, session_id: str) -> None:
        """Delete a specific memory event.
        
        Args:
            event_id: Event ID to delete
            session_id: Session ID in which the event was created
            
        Raises:
            AgentCoreMemoryError: If event deletion fails
        """
        try:
            client = self.aws_manager.get_agentcore_client()
            
            response = client.delete_event(
                memoryId=self.memory_id,
                eventId=event_id,
                actorId=self.actor_id,
                sessionId=session_id
            )
            
            logger.info(f"Deleted event {event_id} from memory {self.memory_id}")
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'delete_event',
                'memory_id': self.memory_id,
                'event_id': event_id
            })
            raise handle_error_gracefully(e, "delete memory event")
    
    def delete_memory_record(self, memory_record_id: str) -> None:
        """Delete a specific memory record.
        
        Args:
            memory_record_id: ID of the memory record to delete
            
        Raises:
            AgentCoreMemoryError: If memory record deletion fails
        """
        try:
            client = self.aws_manager.get_agentcore_client()
            
            logger.info(f"Deleting memoryRecordId {memory_record_id} from memory provider {self.memory_id}")
            
            response = client.delete_memory_record(
                memoryId=self.memory_id,
                memoryRecordId=memory_record_id
            )
            
            logger.info(f"Deleted memory record {memory_record_id}")
            
        except (ClientError, BotoCoreError) as e:
            log_aws_error(e, {
                'operation': 'delete_memory_record',
                'memory_id': self.memory_id,
                'memory_record_id': memory_record_id
            })
            raise handle_error_gracefully(e, "delete memory record")