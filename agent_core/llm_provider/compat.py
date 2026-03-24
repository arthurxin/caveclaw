from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from agent_core.assistant_messages.types import AgentTool
from .provider_types import Model


def supports_store(model: Model) -> bool:
    compat = model.compat
    if compat and compat.supportsStore is not None:
        return bool(compat.supportsStore)
    return False


def supports_developer_role(model: Model) -> bool:
    compat = model.compat
    if compat and compat.supportsDeveloperRole is not None:
        return bool(compat.supportsDeveloperRole)
    return True


def supports_strict_tool_schema(model: Model) -> bool:
    compat = model.compat
    if compat and compat.supportsStrictToolSchema is not None:
        return bool(compat.supportsStrictToolSchema)
    return False


def supports_reasoning_effort(model: Model) -> bool:
    compat = model.compat
    if compat and compat.supportsReasoningEffort is not None:
        return bool(compat.supportsReasoningEffort)
    return bool(model.reasoning)


def supports_usage_in_streaming(model: Model) -> bool:
    compat = model.compat
    if compat and compat.supportsUsageInStreaming is not None:
        return bool(compat.supportsUsageInStreaming)
    return True


def requires_thinking_as_text(model: Model) -> bool:
    compat = model.compat
    if compat and compat.requiresThinkingAsText is not None:
        return bool(compat.requiresThinkingAsText)
    return False


def requires_tool_result_name(model: Model) -> bool:
    compat = model.compat
    if compat and compat.requiresToolResultName is not None:
        return bool(compat.requiresToolResultName)
    return False


def requires_assistant_after_tool_result(model: Model) -> bool:
    compat = model.compat
    if compat and compat.requiresAssistantAfterToolResult is not None:
        return bool(compat.requiresAssistantAfterToolResult)
    return False


def resolve_max_tokens_field(model: Model, default: str = "max_tokens") -> str:
    compat = model.compat
    if compat and compat.maxTokensField:
        return compat.maxTokensField
    return default


def build_reasoning_payload(model: Model, thinking_level: Optional[str]) -> Dict[str, Any]:
    if not thinking_level or thinking_level == "off":
        return {}

    compat = model.compat
    thinking_format = compat.thinkingFormat if compat and compat.thinkingFormat else "openai"

    if thinking_format == "qwen":
        return {"enable_thinking": thinking_level in {"minimal", "low", "medium", "high", "xhigh"}}
    if thinking_format == "zai":
        return {"thinking": {"type": "enabled"}} if thinking_level in {"minimal", "low", "medium", "high", "xhigh"} else {}
    if not supports_reasoning_effort(model):
        return {}

    if thinking_level in {"high", "xhigh"}:
        return {"reasoning_effort": "high"}
    if thinking_level == "medium":
        return {"reasoning_effort": "medium"}
    if thinking_level == "low":
        return {"reasoning_effort": "low"}
    if thinking_level == "minimal":
        return {"reasoning_effort": "minimal"}
    return {}


def normalize_openai_compatible_messages(model: Model, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    previous_role: Optional[str] = None

    for message in messages:
        current = dict(message)
        role = current.get("role")

        if role == "developer" and not supports_developer_role(model):
            current["role"] = "system"
            role = "system"

        if requires_assistant_after_tool_result(model) and previous_role == "tool" and role == "user":
            normalized.append({"role": "assistant", "content": ""})

        if requires_tool_result_name(model) and role == "tool" and not current.get("name"):
            current["name"] = current.get("tool_name") or "tool"

        normalized.append(current)
        previous_role = role if isinstance(role, str) else previous_role

    return normalized


def build_openai_compatible_tools(model: Model, tools: Optional[Iterable[AgentTool]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None

    strict = supports_strict_tool_schema(model)
    payloads: List[Dict[str, Any]] = []
    for tool in tools:
        function_payload: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": _normalize_openai_tool_schema(tool.parameters, strict=strict),
        }
        if strict:
            function_payload["strict"] = True
        payloads.append({"type": "function", "function": function_payload})
    return payloads


def _normalize_openai_tool_schema(schema: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
    normalized = deepcopy(schema)
    if not strict:
        return normalized
    return _ensure_strict_json_schema(normalized)


def _ensure_strict_json_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    normalized = {key: _ensure_strict_json_schema(value) for key, value in schema.items()}
    schema_type = normalized.get("type")

    if schema_type == "object" or (
        schema_type is None and isinstance(normalized.get("properties"), dict)
    ):
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                key: _ensure_strict_json_schema(value)
                for key, value in properties.items()
            }
            normalized.setdefault("additionalProperties", False)

    if schema_type == "array" and isinstance(normalized.get("items"), dict):
        normalized["items"] = _ensure_strict_json_schema(normalized["items"])

    for keyword in ("anyOf", "oneOf", "allOf"):
        variants = normalized.get(keyword)
        if isinstance(variants, list):
            normalized[keyword] = [_ensure_strict_json_schema(value) for value in variants]

    return normalized


__all__ = [
    "build_openai_compatible_tools",
    "build_reasoning_payload",
    "normalize_openai_compatible_messages",
    "requires_assistant_after_tool_result",
    "requires_thinking_as_text",
    "requires_tool_result_name",
    "resolve_max_tokens_field",
    "supports_developer_role",
    "supports_reasoning_effort",
    "supports_store",
    "supports_strict_tool_schema",
    "supports_usage_in_streaming",
]
