from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol, AsyncGenerator, Dict, Any, List, Optional, Awaitable, Callable, Literal

from .provider_types import Model

# Importing engine types directly to interface with them
from agent_core.assistant_messages.types import AgentMessage, AgentTool, Message

Transport = Literal["sse", "websocket", "auto"]
CacheRetention = Literal["none", "short", "long"]
PayloadHook = Callable[[object, Model], object | None | Awaitable[object | None]]


@dataclass
class StreamOptions:
    """Options for generating a response stream."""
    thinking_level: str = "off"
    system_prompt: Optional[str] = None
    tools: Optional[List[AgentTool]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    signal: Optional[Any] = None
    api_key: Optional[str] = None
    transport: Optional[Transport] = None
    cache_retention: Optional[CacheRetention] = None
    session_id: Optional[str] = None
    on_payload: Optional[PayloadHook] = None
    headers: Optional[Dict[str, str]] = None
    max_retry_delay_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    thinking_budgets: Optional[Dict[str, Any]] = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    reasoning: Optional[str] = None


async def maybe_override_payload(payload: object, model: Model, options: StreamOptions) -> object:
    if options.on_payload is None:
        return payload
    replacement = options.on_payload(payload, model)
    if inspect.isawaitable(replacement):
        replacement = await replacement
    return payload if replacement is None else replacement


def merge_request_headers(model: Model, options: StreamOptions) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if model.headers:
        headers.update(model.headers)
    if options.headers:
        headers.update(options.headers)
    return headers


def resolve_effective_max_tokens(model: Model, options: StreamOptions) -> int:
    return int(options.max_tokens if options.max_tokens is not None else model.maxTokens)

class ApiProvider(Protocol):
    """
    The Universal Streamer facade. 
    Specific implementations (e.g., OpenAiProvider, AnthropicProvider) will adapt their
    native SDKs or HTTP requests to yield standard python dictionaries (or AssistantMessage chunks)
    that `agent_loop._stream_assistant_response` can natively consume.
    """
    api: str # Identifies the provider type, e.g., "openai-chat", "anthropic-messages"
    
    async def stream(
        self, 
        model: Model, 
        messages: List[Message], 
        options: StreamOptions,
        api_key: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yields normalized fragments representing the assistant's response.
        Expected format matches what the core engine loop expects:
        - {"content": "text fragment"}
        - {"tool_calls": [{"id": "...", "name": "...", "arguments": "fragment"}]}
        """
        ...
        yield {} # just for type consistency in Protocol

class ApiRegistry:
    """Singleton registry holding all active ApiProviders mapped by their api string."""
    def __init__(self):
        self._providers: Dict[str, ApiProvider] = {}
        
    def register(self, provider: ApiProvider):
        self._providers[provider.api] = provider
        
    def get(self, api: str) -> Optional[ApiProvider]:
        return self._providers.get(api)

    def list(self) -> List[ApiProvider]:
        return list(self._providers.values())

    def unregister(self, api: str) -> None:
        self._providers.pop(api, None)

    def clear(self) -> None:
        self._providers.clear()

# Global singleton
api_provider_registry = ApiRegistry()
