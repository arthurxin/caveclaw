from __future__ import annotations

import os
from typing import Dict

from agent_core.llm_provider.api_registry import api_provider_registry
from agent_core.llm_provider.providers import AnthropicProvider, ArkProvider, AzureProvider, GoogleProvider, MiniMaxProvider, OpenAiProvider


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_JSON_PATH = os.path.join(PROJECT_ROOT, "models.json")
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def build_provider(provider_name: str):
    providers: Dict[str, object] = {
        "openai": OpenAiProvider(),
        "anthropic": AnthropicProvider(),
        "google": GoogleProvider(),
        "minimax": MiniMaxProvider(),
        "volcengine": ArkProvider(),
        "azure": AzureProvider(),
    }
    return providers[provider_name]


def register_demo_providers() -> None:
    for provider_name in ("openai", "anthropic", "azure", "google", "minimax", "volcengine"):
        api_provider_registry.register(build_provider(provider_name))


def default_demo_model_uri() -> str:
    provider = os.environ.get("CAVECLAW_DEFAULT_PROVIDER")
    model = os.environ.get("CAVECLAW_DEFAULT_MODEL")
    if provider and model:
        return f"{provider}/{model}"
    if provider == "azure":
        return "azure/gpt-5.4"
    if os.environ.get("AZURE_API_KEY"):
        return "azure/gpt-5.4"
    return "minimax/minimax"
