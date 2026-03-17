"""
Universal LLM Wrapper Module.
Provides dynamic model registration, resolution, and unified streaming interfaces.
"""

from .provider_types import Model, ProviderConfig, CostConfig, ModelCompat, RoutingPreferences
from .registry import ModelRegistry
from .resolver import ModelResolver, ScopedModel, ParsedModelResult
from .api_registry import ApiProvider, api_provider_registry, StreamOptions

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
    "api_provider_registry",
    "StreamOptions",
]
