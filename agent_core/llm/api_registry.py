from typing import Protocol, AsyncGenerator, Dict, Any, List, Optional
from .provider_types import Model

# Importing engine types directly to interface with them
from agent_core.types import AgentMessage, Message, AgentTool

class StreamOptions:
    """Options for generating a response stream."""
    thinking_level: str = "off"
    system_prompt: Optional[str] = None
    tools: Optional[List[AgentTool]] = None

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

# Global singleton
api_provider_registry = ApiRegistry()
