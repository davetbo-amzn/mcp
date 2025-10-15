#!/usr/bin/env python3
"""Test script for settings file write methods."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add the package to path
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'awslabs'))

from amazon_bedrock_agentcore_mcp_server.memory.config import MemoryConfig


def test_save_to_settings_file_creates_new_file():
    """Test save_to_settings_file creates a new file."""
    print("\n=== Testing save_to_settings_file creates new file ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        # Create config
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2",
            memory_strategy_semantic="semantic-456",
            memory_strategy_summarization="summarization-789"
        )
        
        # Save to new file
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path),
            created_at="2025-10-14T12:00:00Z"
        )
        
        # Verify file was created
        assert temp_path.exists(), "Settings file should be created"
        
        # Verify content
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert "memory_providers" in data
        assert "my-memory" in data["memory_providers"]
        
        provider = data["memory_providers"]["my-memory"]
        assert provider["memory_id"] == "test-memory-123"
        assert provider["strategies"]["semantic"] == "semantic-456"
        assert provider["strategies"]["summarization"] == "summarization-789"
        assert provider["created_at"] == "2025-10-14T12:00:00Z"
        
        print("✓ save_to_settings_file creates new file successfully")


def test_save_to_settings_file_updates_existing():
    """Test save_to_settings_file updates existing file."""
    print("\n=== Testing save_to_settings_file updates existing file ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        # Create initial settings file
        initial_data = {
            "disable_memory": False,
            "memory_providers": {
                "existing-memory": {
                    "memory_id": "existing-123",
                    "strategies": {"semantic": "existing-semantic"},
                    "created_at": "2025-10-13T12:00:00Z"
                }
            }
        }
        
        with open(temp_path, 'w') as f:
            json.dump(initial_data, f)
        
        # Create new config
        config = MemoryConfig(
            memory_id="new-memory-456",
            aws_region="us-east-1",
            memory_strategy_user_preferences="preferences-789"
        )
        
        # Save to existing file
        config.save_to_settings_file(
            memory_name="new-memory",
            file_path=str(temp_path),
            created_at="2025-10-14T13:00:00Z"
        )
        
        # Verify both providers exist
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert len(data["memory_providers"]) == 2, "Should have 2 providers"
        assert "existing-memory" in data["memory_providers"]
        assert "new-memory" in data["memory_providers"]
        
        # Verify existing provider unchanged
        existing = data["memory_providers"]["existing-memory"]
        assert existing["memory_id"] == "existing-123"
        assert existing["strategies"]["semantic"] == "existing-semantic"
        
        # Verify new provider added
        new = data["memory_providers"]["new-memory"]
        assert new["memory_id"] == "new-memory-456"
        assert new["strategies"]["user_preferences"] == "preferences-789"
        
        print("✓ save_to_settings_file preserves existing providers")


def test_save_to_settings_file_prevents_duplicate_names():
    """Test save_to_settings_file prevents duplicate memory names."""
    print("\n=== Testing save_to_settings_file prevents duplicates ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        # Create initial settings file
        initial_data = {
            "disable_memory": False,
            "memory_providers": {
                "my-memory": {
                    "memory_id": "existing-123",
                    "strategies": {},
                    "created_at": "2025-10-13T12:00:00Z"
                }
            }
        }
        
        with open(temp_path, 'w') as f:
            json.dump(initial_data, f)
        
        # Try to save with same name
        config = MemoryConfig(
            memory_id="new-memory-456",
            aws_region="us-west-2"
        )
        
        try:
            config.save_to_settings_file(
                memory_name="my-memory",
                file_path=str(temp_path)
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already exists" in str(e)
            print("✓ save_to_settings_file prevents duplicate names")


def test_save_to_settings_file_handles_corrupted_file():
    """Test save_to_settings_file handles corrupted existing file."""
    print("\n=== Testing save_to_settings_file handles corrupted file ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        # Create corrupted file
        with open(temp_path, 'w') as f:
            f.write("{ invalid json }")
        
        # Try to save
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2"
        )
        
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path)
        )
        
        # Verify new file was created
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert "memory_providers" in data
        assert "my-memory" in data["memory_providers"]
        
        print("✓ save_to_settings_file handles corrupted file gracefully")


def test_save_to_settings_file_creates_parent_directory():
    """Test save_to_settings_file creates parent directory if needed."""
    print("\n=== Testing save_to_settings_file creates parent directory ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "subdir" / "test_settings.json"
        
        # Parent directory doesn't exist yet
        assert not temp_path.parent.exists()
        
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2"
        )
        
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path)
        )
        
        # Verify directory and file were created
        assert temp_path.parent.exists()
        assert temp_path.exists()
        
        print("✓ save_to_settings_file creates parent directory")


def test_save_to_settings_file_with_all_strategies():
    """Test save_to_settings_file with all strategy types."""
    print("\n=== Testing save_to_settings_file with all strategies ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2",
            memory_strategy_semantic="semantic-456",
            memory_strategy_summarization="summarization-789",
            memory_strategy_user_preferences="preferences-012"
        )
        
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path)
        )
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        strategies = data["memory_providers"]["my-memory"]["strategies"]
        assert len(strategies) == 3
        assert strategies["semantic"] == "semantic-456"
        assert strategies["summarization"] == "summarization-789"
        assert strategies["user_preferences"] == "preferences-012"
        
        print("✓ save_to_settings_file saves all strategy types")


def test_save_to_settings_file_with_partial_strategies():
    """Test save_to_settings_file with only some strategies."""
    print("\n=== Testing save_to_settings_file with partial strategies ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2",
            memory_strategy_semantic="semantic-456"
            # Only semantic strategy configured
        )
        
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path)
        )
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        strategies = data["memory_providers"]["my-memory"]["strategies"]
        assert len(strategies) == 1
        assert strategies["semantic"] == "semantic-456"
        assert "summarization" not in strategies
        assert "user_preferences" not in strategies
        
        print("✓ save_to_settings_file handles partial strategies")


def test_save_to_settings_file_default_timestamp():
    """Test save_to_settings_file generates timestamp if not provided."""
    print("\n=== Testing save_to_settings_file default timestamp ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_settings.json"
        
        config = MemoryConfig(
            memory_id="test-memory-123",
            aws_region="us-west-2"
        )
        
        # Don't provide created_at
        config.save_to_settings_file(
            memory_name="my-memory",
            file_path=str(temp_path)
        )
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        created_at = data["memory_providers"]["my-memory"]["created_at"]
        assert created_at is not None
        
        # Verify it's a valid ISO 8601 timestamp
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        print("✓ save_to_settings_file generates default timestamp")


if __name__ == '__main__':
    print("Testing settings file write methods...")
    
    try:
        test_save_to_settings_file_creates_new_file()
        test_save_to_settings_file_updates_existing()
        test_save_to_settings_file_prevents_duplicate_names()
        test_save_to_settings_file_handles_corrupted_file()
        test_save_to_settings_file_creates_parent_directory()
        test_save_to_settings_file_with_all_strategies()
        test_save_to_settings_file_with_partial_strategies()
        test_save_to_settings_file_default_timestamp()
        
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
