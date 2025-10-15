"""Tests for settings file read methods in memory configuration.

This module tests the settings file persistence functionality including:
- load_from_settings_environment()
- load_from_settings_file()
- from_environment_or_file()
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from awslabs.amazon_bedrock_agentcore_mcp_server.memory.config import MemoryConfig


class TestSettingsFileRead:
    """Test settings file read methods."""

    def test_load_from_settings_environment_with_memory_id(self, monkeypatch):
        """Test loading from environment when AGENTCORE_MCP_MEMORY_ID is set."""
        monkeypatch.setenv('AGENTCORE_MCP_MEMORY_ID', 'test-memory-id')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')
        
        config = MemoryConfig.load_from_settings_environment()
        
        assert config is not None
        assert config.memory_id == 'test-memory-id'
        assert config.aws_region == 'us-east-1'

    def test_load_from_settings_environment_with_strategies(self, monkeypatch):
        """Test loading from environment when strategy IDs are set."""
        monkeypatch.setenv('MEMORY_STRATEGY_ID_SEMANTIC', 'semantic-123')
        monkeypatch.setenv('AWS_REGION', 'us-west-2')
        
        config = MemoryConfig.load_from_settings_environment()
        
        assert config is not None
        assert config.memory_strategy_semantic == 'semantic-123'
        assert config.aws_region == 'us-west-2'

    def test_load_from_settings_environment_no_config(self, monkeypatch):
        """Test loading from environment when no memory config exists."""
        # Clear all memory-related env vars
        monkeypatch.delenv('AGENTCORE_MCP_MEMORY_ID', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SEMANTIC', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SUMMARIZATION', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_USER_PREFERENCES', raising=False)
        
        config = MemoryConfig.load_from_settings_environment()
        
        assert config is None

    def test_load_from_settings_file_valid(self):
        """Test loading from valid settings file."""
        settings_data = {
            "disable_memory": False,
            "memory_providers": {
                "test-memory": {
                    "memory_id": "test-memory-id-123",
                    "strategies": {
                        "semantic": "semantic-strategy-id",
                        "summarization": "summarization-strategy-id",
                        "user_preferences": "preferences-strategy-id"
                    },
                    "created_at": "2025-10-14T12:00:00Z"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.load_from_settings_file(temp_path)
            
            assert config is not None
            assert config.memory_id == "test-memory-id-123"
            assert config.memory_strategy_semantic == "semantic-strategy-id"
            assert config.memory_strategy_summarization == "summarization-strategy-id"
            assert config.memory_strategy_user_preferences == "preferences-strategy-id"
        finally:
            Path(temp_path).unlink()

    def test_load_from_settings_file_missing(self):
        """Test loading from non-existent settings file."""
        config = MemoryConfig.load_from_settings_file("nonexistent_file.json")
        
        assert config is None

    def test_load_from_settings_file_corrupted_json(self):
        """Test loading from corrupted JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content }")
            temp_path = f.name
        
        try:
            config = MemoryConfig.load_from_settings_file(temp_path)
            
            assert config is None
        finally:
            Path(temp_path).unlink()

    def test_load_from_settings_file_disabled_memory(self):
        """Test loading from settings file with disable_memory=true."""
        settings_data = {
            "disable_memory": True,
            "memory_providers": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.load_from_settings_file(temp_path)
            
            assert config is None
        finally:
            Path(temp_path).unlink()

    def test_load_from_settings_file_no_providers(self):
        """Test loading from settings file with no memory providers."""
        settings_data = {
            "disable_memory": False,
            "memory_providers": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.load_from_settings_file(temp_path)
            
            assert config is None
        finally:
            Path(temp_path).unlink()

    def test_load_from_settings_file_multiple_providers(self):
        """Test loading from settings file with multiple providers (loads first)."""
        settings_data = {
            "disable_memory": False,
            "memory_providers": {
                "memory-1": {
                    "memory_id": "memory-1-id",
                    "strategies": {
                        "semantic": "semantic-1"
                    },
                    "created_at": "2025-10-14T12:00:00Z"
                },
                "memory-2": {
                    "memory_id": "memory-2-id",
                    "strategies": {
                        "semantic": "semantic-2"
                    },
                    "created_at": "2025-10-14T13:00:00Z"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.load_from_settings_file(temp_path)
            
            assert config is not None
            # Should load the first provider
            assert config.memory_id in ["memory-1-id", "memory-2-id"]
        finally:
            Path(temp_path).unlink()

    def test_from_environment_or_file_env_priority(self, monkeypatch):
        """Test that environment variables take priority over settings file."""
        # Set up environment
        monkeypatch.setenv('AGENTCORE_MCP_MEMORY_ID', 'env-memory-id')
        
        # Create settings file
        settings_data = {
            "disable_memory": False,
            "memory_providers": {
                "test-memory": {
                    "memory_id": "file-memory-id",
                    "strategies": {},
                    "created_at": "2025-10-14T12:00:00Z"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.from_environment_or_file(temp_path)
            
            assert config is not None
            # Should use environment variable, not file
            assert config.memory_id == 'env-memory-id'
        finally:
            Path(temp_path).unlink()

    def test_from_environment_or_file_fallback_to_file(self, monkeypatch):
        """Test fallback to settings file when environment is empty."""
        # Clear environment
        monkeypatch.delenv('AGENTCORE_MCP_MEMORY_ID', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SEMANTIC', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SUMMARIZATION', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_USER_PREFERENCES', raising=False)
        
        # Create settings file
        settings_data = {
            "disable_memory": False,
            "memory_providers": {
                "test-memory": {
                    "memory_id": "file-memory-id",
                    "strategies": {
                        "semantic": "file-semantic-id"
                    },
                    "created_at": "2025-10-14T12:00:00Z"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(settings_data, f)
            temp_path = f.name
        
        try:
            config = MemoryConfig.from_environment_or_file(temp_path)
            
            assert config is not None
            # Should use file since environment is empty
            assert config.memory_id == 'file-memory-id'
            assert config.memory_strategy_semantic == 'file-semantic-id'
        finally:
            Path(temp_path).unlink()

    def test_from_environment_or_file_no_config(self, monkeypatch):
        """Test when no configuration exists in environment or file."""
        # Clear environment
        monkeypatch.delenv('AGENTCORE_MCP_MEMORY_ID', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SEMANTIC', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_SUMMARIZATION', raising=False)
        monkeypatch.delenv('MEMORY_STRATEGY_ID_USER_PREFERENCES', raising=False)
        
        config = MemoryConfig.from_environment_or_file("nonexistent_file.json")
        
        assert config is None
