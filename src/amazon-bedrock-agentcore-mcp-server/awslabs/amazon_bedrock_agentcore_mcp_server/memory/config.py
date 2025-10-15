"""Memory configuration for AgentCore MCP Server.

This module provides configuration management for memory functionality,
including environment variable loading and validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from .types import MemorySettings

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
    
    @classmethod
    def load_from_settings_environment(cls) -> Optional['MemoryConfig']:
        """Load memory configuration from environment variables.
        
        This is a wrapper around from_environment() that returns None
        if no memory configuration is found in environment variables.
        
        Returns:
            MemoryConfig instance if environment variables are set, None otherwise
        """
        config = cls.from_environment()
        
        # Check if any memory configuration exists in environment
        if config.memory_id or config.has_memory_strategies():
            logger.info("Memory configuration found in environment variables")
            return config
        
        logger.debug("No memory configuration found in environment variables")
        return None
    
    @classmethod
    def load_from_settings_file(cls, file_path: str = "memory_settings.json") -> Optional['MemoryConfig']:
        """Load memory configuration from local settings file.
        
        Reads the settings file and loads the first available memory provider.
        Supports multiple memory providers in the settings file.
        
        Args:
            file_path: Path to the settings file (default: "memory_settings.json")
            
        Returns:
            MemoryConfig instance if settings file exists and contains valid configuration,
            None otherwise
        """
        settings_path = Path(file_path)
        
        # Handle missing file gracefully
        if not settings_path.exists():
            logger.debug(f"Settings file not found: {file_path}")
            return None
        
        try:
            # Read and parse JSON file
            with open(settings_path, 'r') as f:
                settings_data = json.load(f)
            
            # Parse using Pydantic model for validation
            settings = MemorySettings(**settings_data)
            
            # Check if memory is disabled
            if settings.disable_memory:
                logger.info("Memory is disabled in settings file")
                return None
            
            # Check if any memory providers exist
            if not settings.memory_providers:
                logger.warning("No memory providers found in settings file")
                return None
            
            # Load the first available memory provider
            memory_name = next(iter(settings.memory_providers))
            provider = settings.memory_providers[memory_name]
            
            logger.info(f"Loading memory configuration from settings file: {memory_name}")
            
            # Create MemoryConfig from provider settings
            config = cls(
                memory_id=provider.memory_id,
                aws_region=os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-west-2')),
                memory_strategy_user_preferences=provider.strategies.get('user_preferences'),
                memory_strategy_summarization=provider.strategies.get('summarization'),
                memory_strategy_semantic=provider.strategies.get('semantic')
            )
            
            logger.info(f"Loaded memory provider '{memory_name}' from settings file")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse settings file {file_path}: {e}")
            logger.warning("Settings file is corrupted, ignoring it")
            return None
        except Exception as e:
            logger.error(f"Error loading settings file {file_path}: {e}")
            return None
    
    @classmethod
    def from_environment_or_file(cls, file_path: str = "memory_settings.json") -> Optional['MemoryConfig']:
        """Load configuration from environment variables, falling back to settings file.
        
        Configuration priority:
        1. Environment variables (highest priority)
        2. Local settings file
        3. None if neither source has configuration
        
        Args:
            file_path: Path to the settings file (default: "memory_settings.json")
            
        Returns:
            MemoryConfig instance if configuration found, None otherwise
        """
        # Try environment variables first
        config = cls.load_from_settings_environment()
        if config:
            logger.info("Using memory configuration from environment variables")
            return config
        
        # Fall back to settings file
        config = cls.load_from_settings_file(file_path)
        if config:
            logger.info("Using memory configuration from settings file")
            return config
        
        # No configuration found
        logger.info("No memory configuration found in environment or settings file")
        return None
    
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

    def save_to_settings_file(
        self,
        memory_name: str,
        file_path: str = "memory_settings.json",
        created_at: Optional[str] = None
    ) -> None:
        """Save memory configuration to local settings file.
        
        Creates the settings file if it doesn't exist, or updates an existing file.
        Preserves existing memory providers when adding a new one.
        Will not overwrite an existing provider with the same name.
        
        Args:
            memory_name: Name for this memory provider (must be unique)
            file_path: Path to the settings file (default: "memory_settings.json")
            created_at: ISO 8601 timestamp of creation (defaults to current time)
            
        Raises:
            ValueError: If a memory provider with the same name already exists
            PermissionError: If unable to write to the settings file
        """
        from datetime import datetime
        
        settings_path = Path(file_path)
        
        # Load existing settings or create new ones
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                settings = MemorySettings(**settings_data)
                logger.info("Loaded existing settings file for update")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    "Failed to load existing settings file, creating new one: %s",
                    e
                )
                settings = MemorySettings()
        else:
            logger.info("Creating new settings file")
            settings = MemorySettings()
        
        # Check if memory provider name already exists
        if memory_name in settings.memory_providers:
            error_msg = (
                f"Memory provider '{memory_name}' already exists in settings file. "
                "Each memory provider must have a unique name."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Build strategies dictionary from current configuration
        strategies = {}
        if self.memory_strategy_user_preferences:
            strategies['user_preferences'] = self.memory_strategy_user_preferences
        if self.memory_strategy_summarization:
            strategies['summarization'] = self.memory_strategy_summarization
        if self.memory_strategy_semantic:
            strategies['semantic'] = self.memory_strategy_semantic
        
        # Create provider settings
        from .types import MemoryProviderSettings
        provider = MemoryProviderSettings(
            memory_id=self.memory_id or "",
            strategies=strategies,
            created_at=created_at or datetime.utcnow().isoformat()
        )
        
        # Add memory provider
        settings.memory_providers[memory_name] = provider
        
        # Write settings file
        try:
            # Ensure parent directory exists
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings.model_dump(), f, indent=2)
            
            logger.info(
                "Saved memory provider '%s' to settings file: %s",
                memory_name,
                file_path
            )
        except PermissionError as e:
            logger.error(
                "Permission denied writing to settings file %s: %s",
                file_path,
                e
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to write settings file %s: %s",
                file_path,
                e
            )
            raise


# Global memory configuration instance
memory_config = MemoryConfig.from_environment()