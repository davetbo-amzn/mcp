# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for memory configuration functionality."""

import os
from awslabs.amazon_bedrock_agentcore_mcp_server.memory.config import MemoryConfig
from awslabs.amazon_bedrock_agentcore_mcp_server.memory.memory_strategy_config import (
    MemoryStrategyConfig,
    MemoryStrategyType
)


class TestMemoryConfig:
    """Test cases for MemoryConfig functionality."""

    def test_memory_config_default_values(self):
        """Test MemoryConfig initializes with correct default values."""
        # Act
        config = MemoryConfig()

        # Assert
        assert config.memory_id is None
        assert config.aws_region == 'us-west-2'
        assert config.memory_strategy_user_preferences is None
        assert config.memory_strategy_summarization is None
        assert config.memory_strategy_semantic is None

    def test_memory_config_custom_values(self):
        """Test MemoryConfig can be initialized with custom values."""
        # Arrange
        memory_id = "test-memory-123"
        aws_region = "us-east-1"
        user_prefs_strategy = "strategy-user-prefs-456"
        summarization_strategy = "strategy-summary-789"
        semantic_strategy = "strategy-semantic-101"

        # Act
        config = MemoryConfig(
            memory_id=memory_id,
            aws_region=aws_region,
            memory_strategy_user_preferences=user_prefs_strategy,
            memory_strategy_summarization=summarization_strategy,
            memory_strategy_semantic=semantic_strategy
        )

        # Assert
        assert config.memory_id == memory_id
        assert config.aws_region == aws_region
        assert config.memory_strategy_user_preferences == user_prefs_strategy
        assert config.memory_strategy_summarization == summarization_strategy
        assert config.memory_strategy_semantic == semantic_strategy

    def test_from_environment_with_all_variables_set(self):
        """Test from_environment loads all environment variables correctly."""
        # Arrange
        test_env = {
            'AGENTCORE_MCP_MEMORY_ID': 'env-memory-123',
            'AWS_REGION': 'eu-west-1',
            'MEMORY_STRATEGY_ID_USER_PREFERENCES': 'env-user-prefs-456',
            'MEMORY_STRATEGY_ID_SUMMARIZATION': 'env-summary-789',
            'MEMORY_STRATEGY_ID_SEMANTIC': 'env-semantic-101'
        }

        # Save original environment
        original_env = {}
        for key in test_env:
            original_env[key] = os.environ.get(key)
            os.environ[key] = test_env[key]

        try:
            # Act
            config = MemoryConfig.from_environment()

            # Assert
            assert config.memory_id == 'env-memory-123'
            assert config.aws_region == 'eu-west-1'
            assert config.memory_strategy_user_preferences == 'env-user-prefs-456'
            assert config.memory_strategy_summarization == 'env-summary-789'
            assert config.memory_strategy_semantic == 'env-semantic-101'
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_from_environment_with_aws_default_region(self):
        """Test from_environment uses AWS_DEFAULT_REGION when AWS_REGION not set."""
        # Arrange
        original_aws_region = os.environ.get('AWS_REGION')
        original_aws_default_region = os.environ.get('AWS_DEFAULT_REGION')

        # Remove AWS_REGION and set AWS_DEFAULT_REGION
        os.environ.pop('AWS_REGION', None)
        os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-2'

        try:
            # Act
            config = MemoryConfig.from_environment()

            # Assert
            assert config.aws_region == 'ap-southeast-2'
        finally:
            # Restore original environment
            if original_aws_region is not None:
                os.environ['AWS_REGION'] = original_aws_region
            if original_aws_default_region is not None:
                os.environ['AWS_DEFAULT_REGION'] = original_aws_default_region
            else:
                os.environ.pop('AWS_DEFAULT_REGION', None)

    def test_from_environment_with_no_region_variables(self):
        """Test from_environment defaults to us-west-2 when no region variables set."""
        # Arrange
        original_aws_region = os.environ.get('AWS_REGION')
        original_aws_default_region = os.environ.get('AWS_DEFAULT_REGION')

        # Remove both region environment variables
        os.environ.pop('AWS_REGION', None)
        os.environ.pop('AWS_DEFAULT_REGION', None)

        try:
            # Act
            config = MemoryConfig.from_environment()

            # Assert
            assert config.aws_region == 'us-west-2'
        finally:
            # Restore original environment
            if original_aws_region is not None:
                os.environ['AWS_REGION'] = original_aws_region
            if original_aws_default_region is not None:
                os.environ['AWS_DEFAULT_REGION'] = original_aws_default_region

    def test_has_memory_strategies_with_no_strategies(self):
        """Test has_memory_strategies returns False when no strategies configured."""
        # Arrange
        config = MemoryConfig()

        # Act & Assert
        assert config.has_memory_strategies() is False

    def test_has_memory_strategies_with_one_strategy(self):
        """Test has_memory_strategies returns True when one strategy configured."""
        # Arrange
        config = MemoryConfig(memory_strategy_user_preferences="strategy-123")

        # Act & Assert
        assert config.has_memory_strategies() is True

    def test_has_memory_strategies_with_all_strategies(self):
        """Test has_memory_strategies returns True when all strategies configured."""
        # Arrange
        config = MemoryConfig(
            memory_strategy_user_preferences="strategy-user-123",
            memory_strategy_summarization="strategy-summary-456",
            memory_strategy_semantic="strategy-semantic-789"
        )

        # Act & Assert
        assert config.has_memory_strategies() is True

    def test_get_configured_strategies_with_no_strategies(self):
        """Test get_configured_strategies returns empty list when no strategies configured."""
        # Arrange
        config = MemoryConfig()

        # Act
        strategies = config.get_configured_strategies()

        # Assert
        assert not strategies

    def test_get_configured_strategies_with_partial_strategies(self):
        """Test get_configured_strategies returns correct list with partial strategies."""
        # Arrange
        config = MemoryConfig(
            memory_strategy_user_preferences="strategy-user-123",
            memory_strategy_semantic="strategy-semantic-789"
        )

        # Act
        strategies = config.get_configured_strategies()

        # Assert
        assert set(strategies) == {'user_preferences', 'semantic'}
        assert len(strategies) == 2

    def test_get_configured_strategies_with_all_strategies(self):
        """Test get_configured_strategies returns all strategy types when all configured."""
        # Arrange
        config = MemoryConfig(
            memory_strategy_user_preferences="strategy-user-123",
            memory_strategy_summarization="strategy-summary-456",
            memory_strategy_semantic="strategy-semantic-789"
        )

        # Act
        strategies = config.get_configured_strategies()

        # Assert
        assert set(strategies) == {'user_preferences', 'summarization', 'semantic'}
        assert len(strategies) == 3


class TestMemoryStrategyConfig:
    """Test cases for MemoryStrategyConfig functionality."""

    def test_memory_strategy_config_initialization(self):
        """Test MemoryStrategyConfig initializes correctly."""
        # Act
        config = MemoryStrategyConfig()

        # Assert
        assert config is not None

    def test_strategy_env_mapping_contains_all_types(self):
        """Test STRATEGY_ENV_MAPPING contains all strategy types."""
        # Act
        mapping = MemoryStrategyConfig.STRATEGY_ENV_MAPPING

        # Assert
        expected_types = {
            MemoryStrategyType.USER_PREFERENCES,
            MemoryStrategyType.SUMMARIZATION,
            MemoryStrategyType.SEMANTIC
        }
        assert set(mapping.keys()) == expected_types

    def test_strategy_env_mapping_has_correct_env_vars(self):
        """Test STRATEGY_ENV_MAPPING has correct environment variable names."""
        # Act
        mapping = MemoryStrategyConfig.STRATEGY_ENV_MAPPING

        # Assert
        assert mapping[MemoryStrategyType.USER_PREFERENCES] == "MEMORY_STRATEGY_ID_USER_PREFERENCES"
        assert mapping[MemoryStrategyType.SUMMARIZATION] == "MEMORY_STRATEGY_ID_SUMMARIZATION"
        assert mapping[MemoryStrategyType.SEMANTIC] == "MEMORY_STRATEGY_ID_SEMANTIC"

    def test_is_valid_strategy_type_with_valid_types(self):
        """Test is_valid_strategy_type returns True for valid strategy types."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act & Assert
        assert config.is_valid_strategy_type("user_preferences") is True
        assert config.is_valid_strategy_type("summarization") is True
        assert config.is_valid_strategy_type("semantic") is True

    def test_is_valid_strategy_type_with_invalid_types(self):
        """Test is_valid_strategy_type returns False for invalid strategy types."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act & Assert
        assert config.is_valid_strategy_type("invalid_type") is False
        assert config.is_valid_strategy_type("") is False
        assert config.is_valid_strategy_type("user_preference") is False  # Missing 's'

    def test_is_strategy_available_with_configured_strategy(self):
        """Test is_strategy_available returns True when strategy is configured."""
        # Arrange
        original_env = os.environ.get('MEMORY_STRATEGY_ID_USER_PREFERENCES')
        os.environ['MEMORY_STRATEGY_ID_USER_PREFERENCES'] = 'test-strategy-123'

        try:
            config = MemoryStrategyConfig()

            # Act & Assert
            assert config.is_strategy_available("user_preferences") is True
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['MEMORY_STRATEGY_ID_USER_PREFERENCES'] = original_env
            else:
                os.environ.pop('MEMORY_STRATEGY_ID_USER_PREFERENCES', None)

    def test_is_strategy_available_with_unconfigured_strategy(self):
        """Test is_strategy_available returns False when strategy is not configured."""
        # Arrange
        original_env = os.environ.get('MEMORY_STRATEGY_ID_SEMANTIC')
        os.environ.pop('MEMORY_STRATEGY_ID_SEMANTIC', None)

        try:
            config = MemoryStrategyConfig()

            # Act & Assert
            assert config.is_strategy_available("semantic") is False
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['MEMORY_STRATEGY_ID_SEMANTIC'] = original_env

    def test_get_strategy_id_with_configured_strategy(self):
        """Test get_strategy_id returns correct ID for configured strategy."""
        # Arrange
        test_strategy_id = 'test-strategy-456'
        original_env = os.environ.get('MEMORY_STRATEGY_ID_SUMMARIZATION')
        os.environ['MEMORY_STRATEGY_ID_SUMMARIZATION'] = test_strategy_id

        try:
            config = MemoryStrategyConfig()

            # Act
            strategy_id = config.get_strategy_id("summarization")

            # Assert
            assert strategy_id == test_strategy_id
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['MEMORY_STRATEGY_ID_SUMMARIZATION'] = original_env
            else:
                os.environ.pop('MEMORY_STRATEGY_ID_SUMMARIZATION', None)

    def test_get_strategy_id_with_invalid_strategy_type(self):
        """Test get_strategy_id returns None for invalid strategy type."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act
        strategy_id = config.get_strategy_id("invalid_type")

        # Assert
        assert strategy_id is None

    def test_get_available_strategies_with_mixed_configuration(self):
        """Test get_available_strategies returns only configured strategies."""
        # Arrange
        original_env = {}
        test_env = {
            'MEMORY_STRATEGY_ID_USER_PREFERENCES': 'user-strategy-123',
            'MEMORY_STRATEGY_ID_SEMANTIC': 'semantic-strategy-789'
        }

        # Save and set environment variables
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Ensure summarization is not set
        original_env['MEMORY_STRATEGY_ID_SUMMARIZATION'] = os.environ.get(
            'MEMORY_STRATEGY_ID_SUMMARIZATION'
        )
        os.environ.pop('MEMORY_STRATEGY_ID_SUMMARIZATION', None)

        try:
            config = MemoryStrategyConfig()

            # Act
            available_strategies = config.get_available_strategies()

            # Assert
            assert available_strategies == {'user_preferences', 'semantic'}
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_get_env_var_name_with_valid_strategy_types(self):
        """Test get_env_var_name returns correct environment variable names."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act & Assert
        assert (config.get_env_var_name("user_preferences") ==
                "MEMORY_STRATEGY_ID_USER_PREFERENCES")
        assert (config.get_env_var_name("summarization") ==
                "MEMORY_STRATEGY_ID_SUMMARIZATION")
        assert (config.get_env_var_name("semantic") ==
                "MEMORY_STRATEGY_ID_SEMANTIC")

    def test_get_env_var_name_with_invalid_strategy_type(self):
        """Test get_env_var_name returns None for invalid strategy type."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act
        env_var = config.get_env_var_name("invalid_type")

        # Assert
        assert env_var is None

    def test_validate_strategy_request_with_valid_configured_strategy(self):
        """Test validate_strategy_request returns success for valid configured strategy."""
        # Arrange
        original_env = os.environ.get('MEMORY_STRATEGY_ID_USER_PREFERENCES')
        os.environ['MEMORY_STRATEGY_ID_USER_PREFERENCES'] = 'test-strategy-123'

        try:
            config = MemoryStrategyConfig()

            # Act
            is_valid, error_message = config.validate_strategy_request("user_preferences")

            # Assert
            assert is_valid is True
            assert error_message == ""
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['MEMORY_STRATEGY_ID_USER_PREFERENCES'] = original_env
            else:
                os.environ.pop('MEMORY_STRATEGY_ID_USER_PREFERENCES', None)

    def test_validate_strategy_request_with_invalid_strategy_type(self):
        """Test validate_strategy_request returns error for invalid strategy type."""
        # Arrange
        config = MemoryStrategyConfig()

        # Act
        is_valid, error_message = config.validate_strategy_request("invalid_type")

        # Assert
        assert is_valid is False
        assert "Invalid strategy type 'invalid_type'" in error_message
        assert "Valid types:" in error_message

    def test_validate_strategy_request_with_unconfigured_strategy(self):
        """Test validate_strategy_request returns error for unconfigured strategy."""
        # Arrange
        original_env = os.environ.get('MEMORY_STRATEGY_ID_SEMANTIC')
        os.environ.pop('MEMORY_STRATEGY_ID_SEMANTIC', None)

        try:
            config = MemoryStrategyConfig()

            # Act
            is_valid, error_message = config.validate_strategy_request("semantic")

            # Assert
            assert is_valid is False
            assert "Strategy 'semantic' not configured" in error_message
            assert "MEMORY_STRATEGY_ID_SEMANTIC" in error_message
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ['MEMORY_STRATEGY_ID_SEMANTIC'] = original_env