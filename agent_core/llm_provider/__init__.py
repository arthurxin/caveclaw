"""
Universal LLM Wrapper Module.
Provides dynamic model registration, resolution, and unified streaming interfaces.
"""

from .provider_types import Model, ProviderConfig, CostConfig, ModelCompat, RoutingPreferences
from .registry import ModelRegistry
from .resolver import ModelResolver, ScopedModel, ParsedModelResult
from .api_registry import ApiProvider, SimpleStreamOptions, api_provider_registry, StreamOptions
from .env_api_keys import get_env_api_key
from .register_builtins import register_builtin_providers
from .stream import complete, complete_simple, prepare_provider_stream, stream, stream_simple
from .usage import calculate_usage_cost, finalize_usage_payload
from .compat import (
    build_openai_compatible_tools,
    build_reasoning_payload,
    normalize_openai_compatible_messages,
    requires_assistant_after_tool_result,
    requires_thinking_as_text,
    requires_tool_result_name,
    resolve_max_tokens_field,
    supports_developer_role,
    supports_reasoning_effort,
    supports_store,
    supports_strict_tool_schema,
    supports_usage_in_streaming,
)
from .validation import ToolValidationError, normalize_tool_arguments, validate_tool_arguments

__all__ = [
    "Model",
    "ProviderConfig",
    "CostConfig",
    "ModelCompat",
    "RoutingPreferences",
    "ModelRegistry",
    "ModelResolver",
    "ScopedModel",
    "ParsedModelResult",
    "ApiProvider",
    "SimpleStreamOptions",
    "api_provider_registry",
    "StreamOptions",
    "complete",
    "complete_simple",
    "calculate_usage_cost",
    "finalize_usage_payload",
    "prepare_provider_stream",
    "stream",
    "stream_simple",
    "get_env_api_key",
    "register_builtin_providers",
    "ToolValidationError",
    "build_openai_compatible_tools",
    "build_reasoning_payload",
    "normalize_openai_compatible_messages",
    "requires_assistant_after_tool_result",
    "normalize_tool_arguments",
    "requires_thinking_as_text",
    "requires_tool_result_name",
    "resolve_max_tokens_field",
    "supports_developer_role",
    "supports_reasoning_effort",
    "supports_store",
    "supports_strict_tool_schema",
    "supports_usage_in_streaming",
    "validate_tool_arguments",
]
