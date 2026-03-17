import fnmatch
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .provider_types import Model
from .registry import ModelRegistry

DEFAULT_THINKING_LEVEL = "off"
VALID_THINKING_LEVELS = {"off", "low", "medium", "high", "xhigh"}

@dataclass
class ScopedModel:
    model: Model
    thinking_level: Optional[str] = None

@dataclass
class ParsedModelResult:
    model: Optional[Model]
    thinking_level: Optional[str] = None
    warning: Optional[str] = None

# Fallback IDs mapped to default providers.
DEFAULT_MODEL_PER_PROVIDER = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "anthropic": "claude-4-6-sonnet-20241022",
    "openai": "gpt-5.4",
    "google": "gemini-3.1-pro-preview",
    "azure": "gpt-5.4",
    "minimax": "minimax",
    "volcengine": "doubao-seed-2-0-lite-260215",
    "xai": "grok-4-fast-non-reasoning",
    "deepseek": "deepseek-coder",
}

def is_valid_thinking_level(level: str) -> bool:
    return level.lower() in VALID_THINKING_LEVELS

def _try_match_model(pattern: str, available_models: List[Model]) -> Optional[Model]:
    """Tries to exact match or fuzzy match a model pattern from available loaded models."""
    
    # Check "provider/modelId" format
    if "/" in pattern:
        provider, model_id = pattern.split("/", 1)
        for m in available_models:
            if m.provider.lower() == provider.lower() and m.id.lower() == model_id.lower():
                return m
                
    # Check exact ID match
    for m in available_models:
        if m.id.lower() == pattern.lower():
            return m
            
    # Check partial match
    matches = []
    for m in available_models:
        if pattern.lower() in m.id.lower() or (m.name and pattern.lower() in m.name.lower()):
            matches.append(m)
            
    if not matches:
        return None
        
    # Heuristics: prefer sorting to get latest
    matches.sort(key=lambda x: x.id, reverse=True)
    return matches[0]

def parse_model_pattern(pattern: str, available_models: List[Model], allow_fallback: bool = True) -> ParsedModelResult:
    """
    Parses a string like "gemma-2:high" or "openai/gpt-4o".
    Extracts the thinking level and resolves the underlying struct.
    """
    exact_match = _try_match_model(pattern, available_models)
    if exact_match:
        return ParsedModelResult(model=exact_match, thinking_level=None)
        
    last_colon_index = pattern.rfind(":")
    if last_colon_index == -1:
        return ParsedModelResult(model=None)
        
    prefix = pattern[:last_colon_index]
    suffix = pattern[last_colon_index+1:]
    
    if is_valid_thinking_level(suffix):
        # Recursive match with suffix chopped off
        result = parse_model_pattern(prefix, available_models, allow_fallback)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=suffix if not result.warning else None,
                warning=result.warning
            )
        return result
    else:
        if not allow_fallback:
            return ParsedModelResult(model=None)
            
        result = parse_model_pattern(prefix, available_models, allow_fallback)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None,
                warning=f"Invalid thinking level '{suffix}' in pattern '{pattern}'. Using default."
            )
        return result

def build_fallback_model(provider: str, model_id: str, available_models: List[Model]) -> Optional[Model]:
    provider_models = [m for m in available_models if m.provider == provider]
    if not provider_models:
        return None
        
    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    base_model = None
    if default_id:
        base_model = next((m for m in provider_models if m.id == default_id), provider_models[0])
    else:
        base_model = provider_models[0]
        
    # Clone and inject ID
    import copy
    new_model = copy.deepcopy(base_model)
    new_model.id = model_id
    new_model.name = model_id
    return new_model

class ModelResolver:
    """Intelligently routes CLI strings and agent preferences to Model structs."""
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def resolve(self, uri: str) -> Tuple[Optional[Model], str]:
        """
        Resolves "provider/model_id:thinking".
        Returns (Model, thinking_level).
        """
        result = parse_model_pattern(uri, self.registry.get_all())
        model = result.model
        thinking_level = result.thinking_level or DEFAULT_THINKING_LEVEL
        
        # If complete failure, try to invent a fallback if provider prefix is known
        if not model and "/" in uri:
            provider, rest = uri.split("/", 1)
            model_id = rest.split(":")[0]
            suffix = rest.rsplit(":", 1)[1] if ":" in rest else None
            model = build_fallback_model(provider, model_id, self.registry.get_all())
            if suffix and is_valid_thinking_level(suffix):
                thinking_level = suffix
            
        return model, thinking_level

    def find_initial_model(self, cli_model_str: Optional[str], default_provider: Optional[str] = "openai", default_model: Optional[str] = "gpt-5.4") -> Tuple[Optional[Model], str]:
        if cli_model_str:
            model, think = self.resolve(cli_model_str)
            if model:
                return model, think

        env_default_provider = os.environ.get("CAVECLAW_DEFAULT_PROVIDER")
        env_default_model = os.environ.get("CAVECLAW_DEFAULT_MODEL")
        if env_default_provider or env_default_model:
            provider = env_default_provider or default_provider
            model_id = env_default_model or DEFAULT_MODEL_PER_PROVIDER.get(provider or "", default_model or "")
            if provider and model_id:
                found = self.registry.find(provider, model_id)
                if found:
                    return found, DEFAULT_THINKING_LEVEL
                fallback = build_fallback_model(provider, model_id, self.registry.get_all())
                if fallback:
                    return fallback, DEFAULT_THINKING_LEVEL
                
        # Try finding default
        if default_provider and default_model:
            found = self.registry.find(default_provider, default_model)
            if found:
                return found, DEFAULT_THINKING_LEVEL
                
        # Fallback to first available globally
        all_models = self.registry.get_all()
        if all_models:
            return all_models[0], DEFAULT_THINKING_LEVEL
            
        return None, DEFAULT_THINKING_LEVEL
