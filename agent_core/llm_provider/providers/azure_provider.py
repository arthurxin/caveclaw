"""
Azure OpenAI Responses API provider.
Uses the Azure-hosted `/openai/v1/responses` endpoint and adapts the
Responses payload into CaveClaw's unified chunk format, including
multi-turn tool-calling continuations via `previous_response_id`.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

from ..api_registry import StreamOptions, merge_request_headers, maybe_override_payload, resolve_effective_max_tokens
from ..message_codec import AzureMessageCodec
from ..normalized_chunks import (
    content_chunk,
    normalize_usage_payload,
    parse_json_arguments,
    provider_state_chunk,
    reasoning_chunk,
    tool_calls_chunk,
    usage_chunk,
)
from ..provider_types import Model
from ..retry import ensure_retry_delay_within_cap, extract_retry_delay_ms


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
        try:
            from openai import APIStatusError, AsyncOpenAI, RateLimitError
        except ImportError:
            raise ImportError("openai package is required: uv add openai")

        key = api_key
        if not key:
            raise ValueError("No Azure API key provided. Resolve it in the host and pass it via config.get_api_key()/config.api_key.")

        endpoint = (model.baseUrl or os.environ.get("AZURE_BASE_URL") or "").strip()
        if not endpoint:
            raise ValueError("No Azure Responses endpoint configured. Set model.baseUrl or AZURE_BASE_URL.")

        payload = _build_azure_payload(messages, model, options)
        payload = dict(await maybe_override_payload(payload, model, options))
        client = AsyncOpenAI(
            api_key=key if _azure_auth_mode() != "api-key" else "unused",
            base_url=_extract_azure_base_url(endpoint),
            default_headers=_build_azure_headers(key, options, model),
        )

        while True:
            try:
                stream_resp = await client.responses.create(**payload, stream=True)
                break
            except (RateLimitError, APIStatusError) as error:
                status_code = getattr(error, "status_code", None)
                if status_code not in {429, 503}:
                    raise
                response = getattr(error, "response", None)
                headers = getattr(response, "headers", None)
                body = getattr(error, "body", None)
                delay_ms = extract_retry_delay_ms(str(body or error), headers)
                if delay_ms is None:
                    raise
                ensure_retry_delay_within_cap(delay_ms, options.max_retry_delay_ms)
                await asyncio.sleep(delay_ms / 1000)
        async for event in stream_resp:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta" and getattr(event, "delta", None):
                yield content_chunk(event.delta)
                continue

            if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"} and getattr(event, "delta", None):
                yield reasoning_chunk(event.delta)
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    yield tool_calls_chunk(
                        [{
                            "id": getattr(item, "call_id", None) or getattr(item, "id", None) or "azure_call",
                            "name": getattr(item, "name", "") or "",
                            "arguments": parse_json_arguments(getattr(item, "arguments", None)),
                        }]
                    )
                continue

            if event_type == "response.completed":
                response = getattr(event, "response", None)
                provider_state = _extract_completed_provider_state(response)
                if provider_state:
                    yield provider_state_chunk("azure", provider_state)
                usage = _extract_completed_usage(response)
                if usage:
                    yield usage_chunk(usage)


def _build_azure_headers(api_key: str, options: StreamOptions, model: Model) -> Dict[str, str]:
    auth_mode = _azure_auth_mode()
    headers = {"Content-Type": "application/json"}
    if auth_mode == "api-key":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(merge_request_headers(model, options))
    return headers


def _azure_auth_mode() -> str:
    return os.environ.get("AZURE_AUTH_MODE", "bearer").lower()


def _extract_azure_base_url(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/responses"):
        return trimmed[: -len("/responses")]
    return trimmed


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
        "max_output_tokens": resolve_effective_max_tokens(model, options),
    }
    if instructions:
        payload["instructions"] = instructions
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    reasoning_effort = _thinking_level_to_effort(options.thinking_level)
    if model.reasoning or reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort or "medium"}
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if options.metadata:
        payload["metadata"] = dict(options.metadata)
    if options.session_id:
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("session_id", options.session_id)
        payload["metadata"] = metadata

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
    return parse_json_arguments(arguments)


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


def _extract_completed_provider_state(response: Any) -> Optional[Dict[str, Any]]:
    if response is None:
        return None
    payload: Dict[str, Any] = {}
    response_id = getattr(response, "id", None)
    status = getattr(response, "status", None)
    if response_id:
        payload["response_id"] = response_id
    if status:
        payload["status"] = status
    return payload or None


def _extract_completed_usage(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return normalize_usage_payload(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
