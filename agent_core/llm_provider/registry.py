import json
import os
from typing import Any, Dict, List, Optional

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
        self.validation_errors: List[str] = []
        
        self.models_json_path = models_json_path or os.path.join(os.getcwd(), "models.json")
        self.refresh()

    def refresh(self):
        """Reload models from disk and rebuild dynamic providers."""
        self.models = []
        self.registered_providers = {}
        self.custom_api_keys.clear()
        self.load_error = None
        self.validation_errors = []
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

            if not isinstance(data, dict):
                raise ValueError("Top-level JSON payload must be an object.")

            providers_data = data.get("providers", {})
            if not isinstance(providers_data, dict):
                raise ValueError("`providers` must be an object keyed by provider name.")
            self._parse_and_merge_providers(providers_data)

        except Exception as e:
            self.load_error = f"Failed to load models.json: {str(e)}"
            return

        if self.validation_errors:
            prefix = "Loaded models.json with validation warnings:"
            self.load_error = "\n".join([prefix, *self.validation_errors])

    def _parse_and_merge_providers(self, providers_data: Dict[str, dict]):
        """Parses the JSON structure and instantiates Model objects."""
        for provider_name, provider_data in providers_data.items():
            if not isinstance(provider_data, dict):
                self._add_validation_error(provider_name, "Provider config must be an object.")
                continue

            # Record api key instructions if provided
            if "apiKey" in provider_data:
                api_key_value = provider_data["apiKey"]
                if isinstance(api_key_value, str) and api_key_value.strip():
                    self.custom_api_keys[provider_name] = api_key_value
                else:
                    self._add_validation_error(provider_name, "`apiKey` must be a non-empty string when provided.")

            base_url = provider_data.get("baseUrl")
            api_type = provider_data.get("api")
            headers = provider_data.get("headers")

            if api_type is None:
                self._add_validation_error(provider_name, "Missing required field `api`.")
                continue
            if not isinstance(api_type, str) or not api_type.strip():
                self._add_validation_error(provider_name, "`api` must be a non-empty string.")
                continue
            if base_url is not None and not isinstance(base_url, str):
                self._add_validation_error(provider_name, "`baseUrl` must be a string when provided.")
                continue
            if headers is not None and not self._is_string_dict(headers):
                self._add_validation_error(provider_name, "`headers` must be an object of string values.")
                continue

            models_list = provider_data.get("models", [])
            if not isinstance(models_list, list):
                self._add_validation_error(provider_name, "`models` must be a list.")
                continue

            self.registered_providers[provider_name] = ProviderConfig(
                baseUrl=base_url,
                apiKey=self.custom_api_keys.get(provider_name),
                api=api_type,
                headers=dict(headers) if headers else None,
                authHeader=provider_data.get("authHeader", True),
                models=[],
            )

            # Construct Models
            for index, m_data in enumerate(models_list):
                if not isinstance(m_data, dict):
                    self._add_validation_error(provider_name, f"`models[{index}]` must be an object.")
                    continue

                model_id = m_data.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    self._add_validation_error(provider_name, f"`models[{index}].id` must be a non-empty string.")
                    continue

                # Merge headers
                model_headers = headers.copy() if headers else {}
                if "headers" in m_data:
                    model_override_headers = m_data["headers"]
                    if not self._is_string_dict(model_override_headers):
                        self._add_validation_error(
                            provider_name,
                            f"`models[{index}].headers` must be an object of string values.",
                        )
                        continue
                    model_headers.update(model_override_headers)

                # Setup CostConfig
                cost = CostConfig()
                if "cost" in m_data:
                    if not isinstance(m_data["cost"], dict):
                        self._add_validation_error(provider_name, f"`models[{index}].cost` must be an object.")
                        continue
                    cost = CostConfig(**m_data["cost"])

                # Setup Compat
                compat = None
                if "compat" in m_data:
                    c_data = m_data["compat"]
                    if not isinstance(c_data, dict):
                        self._add_validation_error(provider_name, f"`models[{index}].compat` must be an object.")
                        continue
                    open_router = RoutingPreferences(**c_data["openRouterRouting"]) if "openRouterRouting" in c_data else None
                    vercel = RoutingPreferences(**c_data["vercelGatewayRouting"]) if "vercelGatewayRouting" in c_data else None
                    compat = ModelCompat(
                        supportsStore=c_data.get("supportsStore"),
                        supportsDeveloperRole=c_data.get("supportsDeveloperRole"),
                        supportsStrictToolSchema=c_data.get("supportsStrictToolSchema"),
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
                    id=model_id,
                    provider=provider_name,
                    name=m_data.get("name", model_id),
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

                self.registered_providers[provider_name].models.append(model)

                # Check authHeader auto-injection from provider standard logic
                if provider_data.get("authHeader", True) and "apiKey" in provider_data:
                    if not model.headers:
                        model.headers = {}
                    # We inject a placeholder or dynamically resolve it. 
                    # For simplicity, we can do it at runtime in Streamer using Registry.get_api_key
                    # but doing it here mirrors pi-mono if resolved.
                    # We will leave it to the runtime streamer to fetch the key to stay safe from leaks.

                self.models.append(model)

    def _add_validation_error(self, provider_name: str, message: str) -> None:
        self.validation_errors.append(f"[provider:{provider_name}] {message}")

    @staticmethod
    def _is_string_dict(value: Any) -> bool:
        return isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items())
