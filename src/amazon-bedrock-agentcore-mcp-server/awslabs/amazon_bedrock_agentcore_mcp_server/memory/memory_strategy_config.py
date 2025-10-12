"""Memory strategy configuration management.

This module provides configuration and validation for memory strategy types,
including mapping between strategy types and environment variables.
"""

import os
import logging
from typing import Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryStrategyType(str, Enum):
    """Supported memory strategy types."""
    USER_PREFERENCES = "user_preferences"
    SUMMARIZATION = "summarization"
    SEMANTIC = "semantic"


class MemoryStrategyConfig:
    """Configuration manager for memory strategies.
    
    Handles strategy type validation, environment variable mapping,
    and strategy availability checking.
    """
    
    # Mapping from strategy types to environment variable names
    STRATEGY_ENV_MAPPING: Dict[str, str] = {
        MemoryStrategyType.USER_PREFERENCES: "MEMORY_STRATEGY_ID_USER_PREFERENCES",
        MemoryStrategyType.SUMMARIZATION: "MEMORY_STRATEGY_ID_SUMMARIZATION",
        MemoryStrategyType.SEMANTIC: "MEMORY_STRATEGY_ID_SEMANTIC",
    }
    
    def __init__(self):
        """Initialize the memory strategy configuration."""
        self._strategy_ids: Dict[str, Optional[str]] = {}
        self._load_strategy_ids()
    
    def _load_strategy_ids(self) -> None:
        """Load strategy IDs from environment variables."""
        for strategy_type, env_var in self.STRATEGY_ENV_MAPPING.items():
            strategy_id = os.getenv(env_var)
            self._strategy_ids[strategy_type] = strategy_id
            
            if strategy_id:
                logger.info(f"Loaded strategy ID for {strategy_type}: {strategy_id}")
            else:
                logger.debug(f"No strategy ID configured for {strategy_type} ({env_var} not set)")
    
    def is_valid_strategy_type(self, strategy_type: str) -> bool:
        """Check if a strategy type is valid.
        
        Args:
            strategy_type: Strategy type to validate
            
        Returns:
            True if the strategy type is valid
        """
        return strategy_type in [e.value for e in MemoryStrategyType]
    
    def is_strategy_available(self, strategy_type: str) -> bool:
        """Check if a strategy is available (configured).
        
        Args:
            strategy_type: Strategy type to check
            
        Returns:
            True if the strategy is configured and available
        """
        if not self.is_valid_strategy_type(strategy_type):
            return False
        return self._strategy_ids.get(strategy_type) is not None
    
    def get_strategy_id(self, strategy_type: str) -> Optional[str]:
        """Get the strategy ID for a given strategy type.
        
        Args:
            strategy_type: Strategy type to get ID for
            
        Returns:
            Strategy ID if configured, None otherwise
        """
        if not self.is_valid_strategy_type(strategy_type):
            logger.warning(f"Invalid strategy type: {strategy_type}")
            return None
        return self._strategy_ids.get(strategy_type)
    
    def get_available_strategies(self) -> Set[str]:
        """Get set of available (configured) strategy types.
        
        Returns:
            Set of strategy type names that are configured
        """
        return {
            strategy_type for strategy_type, strategy_id in self._strategy_ids.items()
            if strategy_id is not None
        }
    
    def get_env_var_name(self, strategy_type: str) -> Optional[str]:
        """Get the environment variable name for a strategy type.
        
        Args:
            strategy_type: Strategy type to get env var for
            
        Returns:
            Environment variable name if valid strategy type, None otherwise
        """
        return self.STRATEGY_ENV_MAPPING.get(strategy_type)
    
    def validate_strategy_request(self, strategy_type: str) -> tuple[bool, str]:
        """Validate a strategy request and provide helpful error message.
        
        Args:
            strategy_type: Strategy type to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_valid_strategy_type(strategy_type):
            valid_types = [e.value for e in MemoryStrategyType]
            return False, f"Invalid strategy type '{strategy_type}'. Valid types: {valid_types}"
        
        if not self.is_strategy_available(strategy_type):
            env_var = self.get_env_var_name(strategy_type)
            return False, f"Strategy '{strategy_type}' not configured. Set environment variable {env_var}"
        
        return True, ""