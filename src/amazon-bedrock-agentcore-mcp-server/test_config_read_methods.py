#!/usr/bin/env python3
"""Test script for settings file read methods."""

import json
import os
import tempfile
from pathlib import Path

# Add the package to path
import sys
from pathlib import Path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'awslabs'))

from amazon_bedrock_agentcore_mcp_server.memory.config import MemoryConfig


def test_load_from_settings_environment():
    """Test load_from_settings_environment method."""
    print("\n=== Testing load_from_settings_environment ===")
    
    # Set environment variables
    os.environ['AGENTCORE_MCP_MEMORY_ID'] = 'test-memory-id'
    os.environ['AWS_REGION'] = 'us-east-1'
    os.environ['MEMORY_STRATEGY_ID_SEMANTIC'] = 'semantic-123'
    
    config = MemoryConfig.load_from_settings_environment()
    
    assert config is not None, "Config should not be None"
    assert config.memory_id == 'test-memory-id', f"Expected 'test-memory-id', got {config.memory_id}"
    assert config.aws_region == 'us-east-1', f"Expected 'us-east-1', got {config.aws_region}"
    assert config.memory_strategy_semantic == 'semantic-123', f"Expected 'semantic-123', got {config.memory_strategy_semantic}"
    
    print("✓ load_from_settings_environment works with environment variables")
    
    # Clear environment
    del os.environ['AGENTCORE_MCP_MEMORY_ID']
    del os.environ['MEMORY_STRATEGY_ID_SEMANTIC']
    
    config = MemoryConfig.load_from_settings_environment()
    assert config is None, "Config should be None when no memory config in environment"
    
    print("✓ load_from_settings_environment returns None when no config")


def test_load_from_settings_file():
    """Test load_from_settings_file method."""
    print("\n=== Testing load_from_settings_file ===")
    
    # Test with valid settings file
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
        
        assert config is not None, "Config should not be None"
        assert config.memory_id == "test-memory-id-123", f"Expected 'test-memory-id-123', got {config.memory_id}"
        assert config.memory_strategy_semantic == "semantic-strategy-id"
        assert config.memory_strategy_summarization == "summarization-strategy-id"
        assert config.memory_strategy_user_preferences == "preferences-strategy-id"
        
        print("✓ load_from_settings_file works with valid settings file")
    finally:
        Path(temp_path).unlink()
    
    # Test with missing file
    config = MemoryConfig.load_from_settings_file("nonexistent_file.json")
    assert config is None, "Config should be None for missing file"
    print("✓ load_from_settings_file handles missing file gracefully")
    
    # Test with corrupted JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name
    
    try:
        config = MemoryConfig.load_from_settings_file(temp_path)
        assert config is None, "Config should be None for corrupted JSON"
        print("✓ load_from_settings_file handles corrupted JSON gracefully")
    finally:
        Path(temp_path).unlink()
    
    # Test with disable_memory=true
    settings_data = {
        "disable_memory": True,
        "memory_providers": {}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(settings_data, f)
        temp_path = f.name
    
    try:
        config = MemoryConfig.load_from_settings_file(temp_path)
        assert config is None, "Config should be None when memory is disabled"
        print("✓ load_from_settings_file respects disable_memory flag")
    finally:
        Path(temp_path).unlink()
    
    # Test with multiple providers
    settings_data = {
        "disable_memory": False,
        "memory_providers": {
            "memory-1": {
                "memory_id": "memory-1-id",
                "strategies": {"semantic": "semantic-1"},
                "created_at": "2025-10-14T12:00:00Z"
            },
            "memory-2": {
                "memory_id": "memory-2-id",
                "strategies": {"semantic": "semantic-2"},
                "created_at": "2025-10-14T13:00:00Z"
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(settings_data, f)
        temp_path = f.name
    
    try:
        config = MemoryConfig.load_from_settings_file(temp_path)
        assert config is not None, "Config should not be None"
        assert config.memory_id in ["memory-1-id", "memory-2-id"], "Should load one of the providers"
        print("✓ load_from_settings_file supports multiple providers")
    finally:
        Path(temp_path).unlink()


def test_from_environment_or_file():
    """Test from_environment_or_file method."""
    print("\n=== Testing from_environment_or_file ===")
    
    # Clear environment first
    for key in ['AGENTCORE_MCP_MEMORY_ID', 'MEMORY_STRATEGY_ID_SEMANTIC', 
                'MEMORY_STRATEGY_ID_SUMMARIZATION', 'MEMORY_STRATEGY_ID_USER_PREFERENCES']:
        if key in os.environ:
            del os.environ[key]
    
    # Test environment priority
    os.environ['AGENTCORE_MCP_MEMORY_ID'] = 'env-memory-id'
    
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
        assert config is not None, "Config should not be None"
        assert config.memory_id == 'env-memory-id', "Should use environment variable, not file"
        print("✓ from_environment_or_file prioritizes environment variables")
    finally:
        Path(temp_path).unlink()
        del os.environ['AGENTCORE_MCP_MEMORY_ID']
    
    # Test fallback to file
    settings_data = {
        "disable_memory": False,
        "memory_providers": {
            "test-memory": {
                "memory_id": "file-memory-id",
                "strategies": {"semantic": "file-semantic-id"},
                "created_at": "2025-10-14T12:00:00Z"
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(settings_data, f)
        temp_path = f.name
    
    try:
        config = MemoryConfig.from_environment_or_file(temp_path)
        assert config is not None, "Config should not be None"
        assert config.memory_id == 'file-memory-id', "Should use file when environment is empty"
        assert config.memory_strategy_semantic == 'file-semantic-id'
        print("✓ from_environment_or_file falls back to settings file")
    finally:
        Path(temp_path).unlink()
    
    # Test no config found
    config = MemoryConfig.from_environment_or_file("nonexistent_file.json")
    assert config is None, "Config should be None when no config found"
    print("✓ from_environment_or_file returns None when no config found")


if __name__ == '__main__':
    print("Testing settings file read methods...")
    
    try:
        test_load_from_settings_environment()
        test_load_from_settings_file()
        test_from_environment_or_file()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
