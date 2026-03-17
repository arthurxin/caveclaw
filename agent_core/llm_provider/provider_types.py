from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Literal

# --- Cost Tracking ---

@dataclass
class CostConfig:
    """Cost settings per 1 million tokens (or operations)."""
    input: float = 0.0
    output: float = 0.0
    cacheRead: float = 0.0
    cacheWrite: float = 0.0

@dataclass
class Usage:
    """Tracks token usage and cost for a specific API call."""
    input: int = 0
    output: int = 0
    cacheRead: int = 0
    cacheWrite: int = 0
    cost: CostConfig = field(default_factory=CostConfig)

# --- Model Compatibility Overrides (Compat) ---

@dataclass
class RoutingPreferences:
    only: Optional[List[str]] = None
    order: Optional[List[str]] = None

@dataclass
class ModelCompat:
    """Optional feature flags and routing compatibility settings for different models."""
    supportsStore: Optional[bool] = None
    supportsDeveloperRole: Optional[bool] = None
    supportsReasoningEffort: Optional[bool] = None
    supportsUsageInStreaming: Optional[bool] = None
    maxTokensField: Optional[Literal["max_completion_tokens", "max_tokens"]] = None
    requiresToolResultName: Optional[bool] = None
    requiresAssistantAfterToolResult: Optional[bool] = None
    requiresThinkingAsText: Optional[bool] = None
    requiresMistralToolIds: Optional[bool] = None
    thinkingFormat: Optional[Literal["openai", "zai", "qwen"]] = None
    openRouterRouting: Optional[RoutingPreferences] = None
    vercelGatewayRouting: Optional[RoutingPreferences] = None

# --- Core Model Definition ---

@dataclass
class Model:
    """
    Metadata representation of an LLM.
    Separated from the core engine's AgentMessage, this defines what the model CAN do and where it lives.
    """
    id: str                                  # e.g., "claude-3-opus"
    provider: str                            # e.g., "anthropic", "openai"
    name: str = ""                           # Human readable name
    api: str = ""                            # e.g., "openai-chat", "anthropic-messages"
    baseUrl: str = ""                        # Root API endpoint
    
    # Capabilities
    reasoning: bool = False
    input: List[Literal["text", "image"]] = field(default_factory=lambda: ["text"])
    contextWindow: int = 128000
    maxTokens: int = 16384
    
    # Billing
    cost: CostConfig = field(default_factory=CostConfig)
    
    # Request overrides
    headers: Optional[Dict[str, str]] = None
    compat: Optional[ModelCompat] = None


# --- Provider Definition ---

@dataclass
class ProviderConfig:
    """
    Configuration for an API Provider (e.g., Anthropic, OpenAI, Local Ollama).
    Typically loaded from models.json or .env overrides.
    """
    baseUrl: Optional[str] = None
    apiKey: Optional[str] = None            # Can be env var name like "OPENAI_API_KEY" or direct string
    api: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    authHeader: Optional[bool] = True
    models: List[Model] = field(default_factory=list)
    
    # Partial overrides for specific models
    modelOverrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
