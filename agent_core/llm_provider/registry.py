import json
import os
from typing import List, Dict, Optional
from dataclasses import asdict
from .provider_types import Model, ProviderConfig, CostConfig, ModelCompat, RoutingPreferences

class ModelRegistry:
    """
    Python port of pi-mono ModelRegistry (packages/coding-agent/src/core/model-registry.ts)
    Manages loading built-in, custom models, and resolving API keys.
    """
    def __init__(self, models_json_path: Optional[str] = None):
        self.models: List[Model] = []
        self.registered_providers: Dict[str, ProviderConfig] = {}
        self.custom_api_keys: Dict[str, str] = {}
        self.load_error: Optional[str] = None
        
        self.models_json_path = models_json_path or os.path.join(os.getcwd(), "models.json")
        self.refresh()

    def refresh(self):
        """Reload models from disk and rebuild dynamic providers."""
        self.custom_api_keys.clear()
        self.load_error = None
        self._load_models()

    def get_all(self) -> List[Model]:
        return self.models

    def find(self, provider: str, model_id: str) -> Optional[Model]:
        for model in self.models:
            if model.provider == provider and model.id == model_id:
                return model
        return None

    def get_api_key(self, provider: str) -> Optional[str]:
        # Try finding dynamically registered ones or overrides from models.json first
        if provider in self.custom_api_keys:
            key_name_or_val = self.custom_api_keys[provider]
            # Try to resolve it from actual env if it looks like a variable name, else direct string
            if key_name_or_val in os.environ:
                return os.environ[key_name_or_val]
            return key_name_or_val
        
        # Fallback to standard environment variables convention
        # e.g. "openai" -> "OPENAI_API_KEY", "anthropic" -> "ANTHROPIC_API_KEY"
        env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
        return os.environ.get(env_var)

    def _load_models(self):
        if not os.path.exists(self.models_json_path):
            self.load_error = f"models.json not found at {self.models_json_path}"
            return

        try:
            with open(self.models_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            providers_data = data.get("providers", {})
            self._parse_and_merge_providers(providers_data)
            
        except Exception as e:
            self.load_error = f"Failed to load models.json: {str(e)}"

    def _parse_and_merge_providers(self, providers_data: Dict[str, dict]):
        """Parses the JSON structure and instantiates Model objects."""
        for provider_name, provider_data in providers_data.items():
            
            # Record api key instructions if provided
            if "apiKey" in provider_data:
                self.custom_api_keys[provider_name] = provider_data["apiKey"]
                
            base_url = provider_data.get("baseUrl")
            api_type = provider_data.get("api")
            headers = provider_data.get("headers")
            
            # Construct Models
            models_list = provider_data.get("models", [])
            for m_data in models_list:
                # Merge headers
                model_headers = headers.copy() if headers else {}
                if "headers" in m_data:
                    model_headers.update(m_data["headers"])
                
                # Setup CostConfig
                cost = CostConfig()
                if "cost" in m_data:
                    cost = CostConfig(**m_data["cost"])
                
                # Setup Compat
                compat = None
                if "compat" in m_data:
                    c_data = m_data["compat"]
                    open_router = RoutingPreferences(**c_data["openRouterRouting"]) if "openRouterRouting" in c_data else None
                    vercel = RoutingPreferences(**c_data["vercelGatewayRouting"]) if "vercelGatewayRouting" in c_data else None
                    compat = ModelCompat(
                        supportsStore=c_data.get("supportsStore"),
                        supportsDeveloperRole=c_data.get("supportsDeveloperRole"),
                        supportsReasoningEffort=c_data.get("supportsReasoningEffort"),
                        supportsUsageInStreaming=c_data.get("supportsUsageInStreaming"),
                        maxTokensField=c_data.get("maxTokensField"),
                        requiresToolResultName=c_data.get("requiresToolResultName"),
                        requiresAssistantAfterToolResult=c_data.get("requiresAssistantAfterToolResult"),
                        requiresThinkingAsText=c_data.get("requiresThinkingAsText"),
                        requiresMistralToolIds=c_data.get("requiresMistralToolIds"),
                        thinkingFormat=c_data.get("thinkingFormat"),
                        openRouterRouting=open_router,
                        vercelGatewayRouting=vercel
                    )
                
                model = Model(
                    id=m_data.get("id", "unknown"),
                    provider=provider_name,
                    name=m_data.get("name", m_data.get("id")),
                    api=m_data.get("api", api_type),
                    baseUrl=m_data.get("baseUrl", base_url),
                    reasoning=m_data.get("reasoning", False),
                    input=m_data.get("input", ["text"]),
                    contextWindow=m_data.get("contextWindow", 128000),
                    maxTokens=m_data.get("maxTokens", 16384),
                    cost=cost,
                    headers=model_headers if model_headers else None,
                    compat=compat
                )
                
                # Check authHeader auto-injection from provider standard logic
                if provider_data.get("authHeader", True) and "apiKey" in provider_data:
                    if not model.headers:
                        model.headers = {}
                    # We inject a placeholder or dynamically resolve it. 
                    # For simplicity, we can do it at runtime in Streamer using Registry.get_api_key
                    # but doing it here mirrors pi-mono if resolved.
                    # We will leave it to the runtime streamer to fetch the key to stay safe from leaks.
                
                self.models.append(model)
