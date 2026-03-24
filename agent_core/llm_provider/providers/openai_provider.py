"""
OpenAI Provider adapter.
Normalizes the OpenAI Streaming API response chunks into the unified dict format
expected by agent_loop.
"""
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions, merge_request_headers, maybe_override_payload, resolve_effective_max_tokens
from ..compat import (
    build_openai_compatible_tools,
    build_reasoning_payload,
    normalize_openai_compatible_messages,
    resolve_max_tokens_field,
    supports_usage_in_streaming,
)
from ..message_codec import OpenAICompatibleMessageCodec
from ..normalized_chunks import content_chunk, normalize_usage_payload, parse_json_arguments, tool_calls_chunk, usage_chunk
from ..retry import ensure_retry_delay_within_cap, extract_retry_delay_ms


class OpenAiProvider:
    """
    Implements the ApiProvider protocol for the OpenAI Completions API.
    Handles both regular text content and streamed tool_calls delta assembly.

    Supports:
    - openai (official api)
    - openai-compatible APIs (DeepSeek, Groq, etc.) via custom baseUrl

    base_url resolution priority:
      1. model.baseUrl (from models.json, if set)
      2. OPENAI_BASE_URL env var
      3. default: None (uses official OpenAI endpoint)
    """
    api = "openai-chat"
    message_codec = OpenAICompatibleMessageCodec()

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
        base_url = model.baseUrl or None
        default_headers = merge_request_headers(model, options) or None

        client = AsyncOpenAI(api_key=key, base_url=base_url, default_headers=default_headers)

        provider_messages = normalize_openai_compatible_messages(model, messages)

        openai_tools = build_openai_compatible_tools(model, options.tools)

        kwargs: Dict[str, Any] = dict(model=model.id, messages=provider_messages)
        kwargs[resolve_max_tokens_field(model)] = resolve_effective_max_tokens(model, options)
        if supports_usage_in_streaming(model):
            kwargs["stream_options"] = {"include_usage": True}
        if openai_tools:
            kwargs["tools"] = openai_tools
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.metadata:
            kwargs["metadata"] = dict(options.metadata)
        if options.session_id:
            metadata = dict(kwargs.get("metadata") or {})
            metadata.setdefault("session_id", options.session_id)
            kwargs["metadata"] = metadata
        kwargs.update(build_reasoning_payload(model, options.thinking_level))
        kwargs = dict(await maybe_override_payload(kwargs, model, options))

        # Accumulator for streaming tool call fragments
        tool_call_accum: Dict[int, Dict[str, Any]] = {}

        while True:
            try:
                stream_resp = await client.chat.completions.create(**kwargs, stream=True)
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

        async for chunk in stream_resp:
            if getattr(chunk, "usage", None):
                usage = normalize_usage_payload(
                    input_tokens=getattr(chunk.usage, "prompt_tokens", None),
                    output_tokens=getattr(chunk.usage, "completion_tokens", None),
                    total_tokens=getattr(chunk.usage, "total_tokens", None),
                )
                if usage:
                    yield usage_chunk(usage)

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text content fragment
            if delta.content:
                yield content_chunk(delta.content)

            # Tool call delta fragments
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_call_accum[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_accum[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

        # After stream ends, emit assembled tool_calls if any
        if tool_call_accum:
            assembled = []
            for idx in sorted(tool_call_accum.keys()):
                tc = tool_call_accum[idx]
                args = parse_json_arguments(tc["arguments"])
                assembled.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": args,
                })
            yield tool_calls_chunk(assembled)
