"""
Azure OpenAI Responses API provider.
Uses the Azure-hosted `/openai/v1/responses` endpoint and adapts the
Responses payload into CaveClaw's unified chunk format, including
multi-turn tool-calling continuations via `previous_response_id`.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

import httpx

from ..api_registry import StreamOptions
from ..message_codec import AzureMessageCodec
from ..provider_types import Model


class AzureProvider:
    api = "azure-responses"
    message_codec = AzureMessageCodec()

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        key = api_key or os.environ.get("AZURE_API_KEY")
        if not key:
            raise ValueError("No Azure API key found. Set AZURE_API_KEY in your .env")

        endpoint = (model.baseUrl or os.environ.get("AZURE_BASE_URL") or "").strip()
        if not endpoint:
            raise ValueError("No Azure Responses endpoint configured. Set model.baseUrl or AZURE_BASE_URL.")

        payload = _build_azure_payload(messages, model, options)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            response = await client.post(
                endpoint,
                headers=_build_azure_headers(key),
                json=payload,
            )

        if response.status_code >= 400:
            raise ValueError(_format_azure_error(response))

        body = response.json()
        provider_state, reasoning, content, tool_calls = _parse_azure_response(body)

        if provider_state:
            yield {"provider_state": {"azure": provider_state}}
        if reasoning:
            yield {"reasoning": reasoning}
        if content:
            yield {"content": content}
        if tool_calls:
            yield {"tool_calls": tool_calls}


def _build_azure_headers(api_key: str) -> Dict[str, str]:
    auth_mode = os.environ.get("AZURE_AUTH_MODE", "bearer").lower()
    headers = {"Content-Type": "application/json"}
    if auth_mode == "api-key":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_azure_payload(
    messages: Sequence[Dict[str, Any]],
    model: Model,
    options: StreamOptions,
) -> Dict[str, Any]:
    previous_response_id, continuation_messages = _find_previous_response_context(messages)

    if previous_response_id is not None:
        instructions = None
        input_items = _messages_to_azure_input(continuation_messages, allow_system=False)
    else:
        instructions, input_items = _messages_to_azure_input(messages, explicit_system_prompt=options.system_prompt)

    payload: Dict[str, Any] = {
        "model": model.id,
        "input": input_items,
        "max_output_tokens": model.maxTokens,
    }
    if instructions:
        payload["instructions"] = instructions
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    reasoning_effort = _thinking_level_to_effort(options.thinking_level)
    if model.reasoning or reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort or "medium"}

    if options.tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in options.tools
        ]
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = True

    return payload


def _find_previous_response_context(
    messages: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        provider_state = (message.get("provider_state") or {}).get("azure", {})
        response_id = provider_state.get("response_id")
        if response_id:
            return str(response_id), list(messages[index + 1 :])
    return None, list(messages)


def _messages_to_azure_input(
    messages: Sequence[Dict[str, Any]],
    explicit_system_prompt: Optional[str] = None,
    *,
    allow_system: bool = True,
) -> Tuple[Optional[str], List[Dict[str, Any]]] | List[Dict[str, Any]]:
    instructions_parts: List[str] = []
    if explicit_system_prompt:
        instructions_parts.append(explicit_system_prompt)

    input_items: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content") or "")

        if role == "system":
            if allow_system and content:
                instructions_parts.append(content)
            continue

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": content,
                }
            )
            continue

        input_items.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )

    if not allow_system:
        return input_items

    instructions = "\n\n".join(part for part in instructions_parts if part) or None
    return instructions, input_items


def _thinking_level_to_effort(thinking_level: str) -> Optional[str]:
    if thinking_level in ("high", "xhigh"):
        return "high"
    if thinking_level == "medium":
        return "medium"
    if thinking_level == "low":
        return "low"
    return None


def _parse_azure_response(
    body: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    provider_state: Dict[str, Any] = {}
    if body.get("id"):
        provider_state["response_id"] = body["id"]
    if body.get("status"):
        provider_state["status"] = body["status"]

    for output_item in body.get("output", []):
        item_type = output_item.get("type")

        if item_type == "reasoning":
            reasoning_parts.extend(_extract_reasoning_parts(output_item))
            continue

        if item_type == "function_call":
            tool_calls.append(
                {
                    "id": output_item.get("call_id") or output_item.get("id") or "azure_call",
                    "name": output_item.get("name", ""),
                    "arguments": _parse_tool_arguments(output_item.get("arguments")),
                }
            )
            continue

        if item_type != "message":
            continue

        for content_item in output_item.get("content", []):
            content_type = content_item.get("type")
            if content_type == "output_text":
                text = _maybe_unescape_text(content_item.get("text"))
                if text:
                    content_parts.append(text)
            elif content_type in ("reasoning_text", "summary_text"):
                text = _maybe_unescape_text(content_item.get("text"))
                if text:
                    reasoning_parts.append(text)

    if not content_parts and body.get("output_text"):
        content_parts.append(_maybe_unescape_text(body.get("output_text")))

    reasoning = "\n".join(part for part in reasoning_parts if part) or None
    content = "\n".join(part for part in content_parts if part) or None
    return provider_state, reasoning, content, tool_calls or None


def _extract_reasoning_parts(output_item: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for summary_item in output_item.get("summary", []):
        if isinstance(summary_item, dict):
            text = _maybe_unescape_text(summary_item.get("text"))
            if text:
                parts.append(text)
    if not parts and isinstance(output_item.get("text"), str):
        text = _maybe_unescape_text(output_item.get("text"))
        if text:
            parts.append(text)
    return parts


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
    return {"raw": arguments}


def _maybe_unescape_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    if "\\u" not in value and "\\n" not in value and '\\"' not in value:
        return value
    try:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return json.loads(f'"{escaped}"')
    except Exception:
        return value


def _format_azure_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return f"Azure Responses request failed ({response.status_code}): {payload}"
