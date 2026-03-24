from __future__ import annotations

from .api_registry import api_provider_registry
from .providers import AnthropicProvider, ArkProvider, AzureProvider, GoogleProvider, MiniMaxProvider, OpenAiProvider


def register_builtin_providers() -> None:
    for provider in (
        OpenAiProvider(),
        AnthropicProvider(),
        GoogleProvider(),
        MiniMaxProvider(),
        ArkProvider(),
        AzureProvider(),
    ):
        api_provider_registry.register(provider)


__all__ = ["register_builtin_providers"]
