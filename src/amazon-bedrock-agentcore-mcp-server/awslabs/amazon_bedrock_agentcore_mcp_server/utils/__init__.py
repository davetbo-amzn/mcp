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

"""Utility modules for AgentCore MCP Server."""

from .aws_client import AWSClientManager, create_aws_client_manager, resolve_aws_region
from .caller_identity import get_actor_id, get_account_id, parse_actor_id_components
from .exceptions import (
    MCPServerError,
    AWSError,
    AgentCoreMemoryError,
    ConfigurationError,
    map_aws_error,
    map_memory_error,
    handle_error_gracefully,
    log_aws_error
)

__all__ = [
    'AWSClientManager',
    'create_aws_client_manager',
    'resolve_aws_region',
    'get_actor_id',
    'get_account_id',
    'parse_actor_id_components',
    'MCPServerError',
    'AWSError',
    'AgentCoreMemoryError',
    'ConfigurationError',
    'map_aws_error',
    'map_memory_error',
    'handle_error_gracefully',
    'log_aws_error'
]
