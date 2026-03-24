"""
Google Gemini Provider adapter.
Uses the Gemini REST streaming API directly so we can preserve provider-specific
parts such as `thoughtSignature` during tool-calling replay.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

import httpx

from ..api_registry import StreamOptions, merge_request_headers, maybe_override_payload, resolve_effective_max_tokens
from ..message_codec import GoogleMessageCodec
from ..normalized_chunks import content_chunk, normalize_usage_payload, provider_state_chunk, reasoning_chunk, tool_calls_chunk, usage_chunk
from ..provider_types import Model
from ..retry import ensure_retry_delay_within_cap, extract_retry_delay_ms

GOOGLE_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider:
    api = "google-gemini"
    message_codec = GoogleMessageCodec()

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        key = api_key
        if not key:
            raise ValueError("No Google API key provided. Resolve it in the host and pass it via config.get_api_key()/config.api_key.")

        base_url = (model.baseUrl or GOOGLE_DEFAULT_BASE_URL).rstrip("/")
        url = f"{base_url}/models/{model.id}:streamGenerateContent"

        payload = _build_generate_content_payload(messages, model, options)
        payload = dict(await maybe_override_payload(payload, model, options))
        emitted_tool_calls: set[str] = set()
        tool_call_counter = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            while True:
                async with client.stream(
                    "POST",
                    url,
                    params={"alt": "sse", "key": key},
                    headers={"Content-Type": "application/json", **merge_request_headers(model, options)},
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        body_text = error_body.decode("utf-8", errors="replace")
                        retry_delay_ms = None
                        if response.status_code in {429, 503}:
                            retry_delay_ms = extract_retry_delay_ms(body_text, response.headers)
                        if retry_delay_ms is not None:
                            ensure_retry_delay_within_cap(retry_delay_ms, options.max_retry_delay_ms)
                            await asyncio.sleep(retry_delay_ms / 1000)
                            continue
                        raise ValueError(_format_google_error(response, error_body))

                    async for event_payload in _iter_sse_json(response):
                        usage = _extract_google_usage(event_payload)
                        if usage:
                            yield usage_chunk(usage)
                        for candidate in event_payload.get("candidates", []):
                            content = candidate.get("content") or {}
                            for part in content.get("parts", []):
                                provider_state = _extract_gemini_provider_state(part)
                                is_thought_part = bool(provider_state and provider_state.get("thought_signatures"))

                                text = part.get("text")
                                if text:
                                    payload_chunk: Dict[str, Any] = (
                                        reasoning_chunk(text) if is_thought_part else content_chunk(text)
                                    )
                                    if provider_state:
                                        payload_chunk.update(provider_state_chunk("gemini", provider_state))
                                    yield payload_chunk
                                    continue

                                function_call = part.get("functionCall")
                                if function_call:
                                    tool_call_key = json.dumps(function_call, ensure_ascii=False, sort_keys=True)
                                    if tool_call_key in emitted_tool_calls:
                                        continue
                                    emitted_tool_calls.add(tool_call_key)
                                    tool_call_counter += 1
                                    payload_chunk = tool_calls_chunk(
                                        [
                                            {
                                                "id": f"gemini_call_{tool_call_counter}",
                                                "name": function_call.get("name", ""),
                                                "arguments": dict(function_call.get("args") or {}),
                                            }
                                        ]
                                    )
                                    if provider_state:
                                        payload_chunk.update(provider_state_chunk("gemini", provider_state))
                                    yield payload_chunk
                                    continue

                                if provider_state:
                                    yield provider_state_chunk("gemini", provider_state)
                    break


def _build_generate_content_payload(
    messages: List[Dict[str, Any]],
    model: Model,
    options: StreamOptions,
) -> Dict[str, Any]:
    system_instruction = _resolve_system_instruction(messages, options.system_prompt)
    history, last_message = _convert_messages(messages)

    contents = list(history)
    if last_message:
        last_role, last_parts = last_message
        contents.append({"role": last_role, "parts": last_parts})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": resolve_effective_max_tokens(model, options),
        },
    }
    if options.temperature is not None:
        payload["generationConfig"]["temperature"] = options.temperature
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if options.tools:
        payload["tools"] = [{"functionDeclarations": [_tool_to_google_schema(tool) for tool in options.tools]}]
    return payload


def _resolve_system_instruction(messages: Sequence[Dict[str, Any]], explicit_system_prompt: Optional[str]) -> Optional[str]:
    system_parts: List[str] = []
    if explicit_system_prompt:
        system_parts.append(explicit_system_prompt)
    for message in messages:
        if message.get("role") == "system" and message.get("content"):
            system_parts.append(str(message["content"]))
    if not system_parts:
        return None
    return "\n\n".join(part for part in system_parts if part)


def _tool_to_google_schema(tool: Any) -> Dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": _convert_schema_to_gemini(tool.parameters),
    }


def _convert_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    unsupported_keys = {"$schema", "additionalProperties", "default", "examples", "const"}
    type_mapping = {
        "object": "OBJECT",
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }

    payload: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in unsupported_keys:
            continue
        if key == "type":
            payload["type"] = type_mapping[str(value).lower()]
        elif key == "properties" and isinstance(value, dict):
            payload["properties"] = {
                prop_key: _convert_schema_to_gemini(prop_schema)
                for prop_key, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            payload["items"] = _convert_schema_to_gemini(value)
        elif key == "required" and isinstance(value, list):
            payload["required"] = list(value)
        elif key == "description":
            payload["description"] = value
        elif key == "enum" and isinstance(value, list):
            payload["enum"] = list(value)
        elif key == "format":
            payload["format"] = value
        elif key == "nullable":
            payload["nullable"] = bool(value)
        elif key == "minItems":
            payload["minItems"] = int(value)
        elif key == "maxItems":
            payload["maxItems"] = int(value)
    return payload


def _convert_messages(
    messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, List[Dict[str, Any]]]]]:
    """
    Convert agent messages into Gemini REST `contents`.
    Returns `(history, last_message)` so callers can preserve the old unit-test
    contract while still building a full contents array.
    """
    history: List[Dict[str, Any]] = []
    last_message: Optional[Tuple[str, List[Dict[str, Any]]]] = None

    conversation = [message for message in messages if message.get("role") != "system"]
    if not conversation:
        return history, last_message

    contents: List[Dict[str, Any]] = []
    for role, parts in (_message_to_google_content(message) for message in conversation):
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": list(parts)})

    if contents and contents[-1]["role"] == "user":
        last_content = contents.pop()
        last_message = (last_content["role"], last_content["parts"])

    history = contents
    return history, last_message


def _message_to_google_content(message: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    role = message.get("role", "user")
    provider_state = (message.get("provider_state") or {}).get("gemini", {})
    provider_parts = provider_state.get("parts")

    gemini_role = "user" if role in ("user", "tool") else "model"
    if isinstance(provider_parts, list) and provider_parts:
        return gemini_role, [_normalize_gemini_part(part) for part in provider_parts]

    content = message.get("content", "")
    return gemini_role, [{"text": content}]


def _normalize_gemini_part(part: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(part)

    if "function_call" in normalized and "functionCall" not in normalized:
        normalized["functionCall"] = normalized.pop("function_call")
    if "function_response" in normalized and "functionResponse" not in normalized:
        normalized["functionResponse"] = normalized.pop("function_response")
    if "thought_signature" in normalized and "thoughtSignature" not in normalized:
        normalized["thoughtSignature"] = normalized.pop("thought_signature")

    function_call = normalized.get("functionCall")
    if isinstance(function_call, dict):
        if "args" not in function_call and isinstance(function_call.get("arguments"), dict):
            function_call = dict(function_call)
            function_call["args"] = function_call.pop("arguments")
            normalized["functionCall"] = function_call

    return normalized


def _extract_gemini_provider_state(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized_part = _normalize_gemini_part(part)
    thought_signature = normalized_part.get("thoughtSignature")
    if not thought_signature:
        return None

    part_payload: Dict[str, Any] = {"thoughtSignature": thought_signature}
    if "text" in normalized_part:
        part_payload["text"] = normalized_part.get("text", "")
    if "functionCall" in normalized_part:
        part_payload["functionCall"] = dict(normalized_part["functionCall"])
    if "functionResponse" in normalized_part:
        part_payload["functionResponse"] = dict(normalized_part["functionResponse"])

    return {
        "parts": [part_payload],
        "thought_signatures": [thought_signature],
    }


def _extract_google_usage(event_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage_metadata = event_payload.get("usageMetadata")
    if not isinstance(usage_metadata, dict):
        return None
    return normalize_usage_payload(
        input_tokens=usage_metadata.get("promptTokenCount"),
        output_tokens=usage_metadata.get("candidatesTokenCount"),
        total_tokens=usage_metadata.get("totalTokenCount"),
        extra={"thoughts_token_count": usage_metadata.get("thoughtsTokenCount")} if "thoughtsTokenCount" in usage_metadata else None,
    )


async def _iter_sse_json(response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    event_lines: List[str] = []

    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line:
            if event_lines:
                payload = "".join(event_lines).strip()
                event_lines = []
                if payload and payload != "[DONE]":
                    yield json.loads(payload)
            continue
        if line.startswith("data:"):
            event_lines.append(line[5:].lstrip())

    if event_lines:
        payload = "".join(event_lines).strip()
        if payload and payload != "[DONE]":
            yield json.loads(payload)


def _format_google_error(response: httpx.Response, body: bytes | None = None) -> str:
    raw_body = body if body is not None else response.content
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        payload = raw_body.decode("utf-8", errors="replace")
    return f"Google Gemini request failed ({response.status_code}): {payload}"
