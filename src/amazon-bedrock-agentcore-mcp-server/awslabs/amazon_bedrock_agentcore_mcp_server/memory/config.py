"""Memory configuration for AgentCore MCP Server.

This module provides configuration management for memory functionality,
including environment variable loading and validation.
"""

import os
import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Configuration settings for memory functionality.
    
    Loads from environment variables following AWS best practices.
    """
    
    memory_id: Optional[str] = Field(default=None, description="AgentCore memory ID")
    aws_region: str = Field(default='us-west-2', description="AWS region for AgentCore services")
    
    # Memory strategy configurations loaded from environment
    memory_strategy_user_preferences: Optional[str] = Field(
        default=None, 
        description="Memory strategy ID for user preferences"
    )
    memory_strategy_summarization: Optional[str] = Field(
        default=None, 
        description="Memory strategy ID for summarization"
    )
    memory_strategy_semantic: Optional[str] = Field(
        default=None, 
        description="Memory strategy ID for semantic memory"
    )
    
    @classmethod
    def from_environment(cls) -> 'MemoryConfig':
        """Load memory configuration from environment variables.
        
        Returns:
            MemoryConfig instance with values loaded from environment
        """
        config = cls(
            memory_id=os.getenv('AGENTCORE_MCP_MEMORY_ID'),
            aws_region=os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-west-2')),
            memory_strategy_user_preferences=os.getenv('MEMORY_STRATEGY_ID_USER_PREFERENCES'),
            memory_strategy_summarization=os.getenv('MEMORY_STRATEGY_ID_SUMMARIZATION'),
            memory_strategy_semantic=os.getenv('MEMORY_STRATEGY_ID_SEMANTIC')
        )
        
        logger.info(f"Loaded memory configuration from environment: region={config.aws_region}")
        if config.memory_id:
            logger.info(f"Memory ID configured: {config.memory_id}")
        else:
            logger.warning("No memory ID configured (AGENTCORE_MCP_MEMORY_ID not set)")
            
        return config
    
    def has_memory_strategies(self) -> bool:
        """Check if any memory strategies are configured.
        
        Returns:
            True if at least one memory strategy is configured
        """
        return any([
            self.memory_strategy_user_preferences,
            self.memory_strategy_summarization,
            self.memory_strategy_semantic
        ])
    
    def get_configured_strategies(self) -> list[str]:
        """Get list of configured memory strategy types.
        
        Returns:
            List of strategy type names that are configured
        """
        strategies = []
        if self.memory_strategy_user_preferences:
            strategies.append('user_preferences')
        if self.memory_strategy_summarization:
            strategies.append('summarization')
        if self.memory_strategy_semantic:
            strategies.append('semantic')
        return strategies


# Global memory configuration instance
memory_config = MemoryConfig.from_environment()