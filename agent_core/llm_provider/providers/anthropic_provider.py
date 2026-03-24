"""
Anthropic Provider adapter.
Normalizes Anthropic's streaming API (content_block_delta events, tool_use blocks)
into the unified dict format expected by agent_loop.

Anthropic's streaming is different from OpenAI:
- Text arrives via content_block_delta events with type="text_delta"
- Tool calls arrive via content_block_delta with type="input_json_delta" (JSON fragments)
- We must track which content block we are in via content_block_start events
"""
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions, merge_request_headers, maybe_override_payload, resolve_effective_max_tokens
from ..message_codec import AnthropicMessageCodec
from ..normalized_chunks import (
    content_chunk,
    normalize_usage_payload,
    parse_json_arguments,
    provider_state_chunk,
    reasoning_chunk,
    tool_calls_chunk,
    usage_chunk,
)


class AnthropicProvider:
    """
    Implements the ApiProvider protocol for the Anthropic Messages API.
    Handles streaming of text and tool_use content block events.

    Supports:
    - anthropic (official api, claude-* models)
    - amazon-bedrock (via boto3 with Anthropic models, handled separately)
    """
    api = "anthropic-messages"
    message_codec = AnthropicMessageCodec()

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required: uv add anthropic")

        key = api_key
        base_url = model.baseUrl

        client_kwargs: Dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url
        extra_headers = merge_request_headers(model, options)
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers

        client = anthropic.AsyncAnthropic(**client_kwargs)

        # Build tools payload if provided
        anthropic_tools = None
        if options.tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in options.tools
            ]

        # Separate system prompt from conversation history
        system_prompt = options.system_prompt
        conversation = [m for m in messages if m.get("role") != "system"]

        kwargs: Dict[str, Any] = dict(
            model=model.id,
            max_tokens=resolve_effective_max_tokens(model, options),
            messages=conversation,
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.metadata:
            kwargs["metadata"] = dict(options.metadata)

        # Thinking (extended thinking) support
        if options.thinking_level in ("high", "xhigh") and model.reasoning:
            budget_tokens = 8000
            if options.thinking_budgets:
                configured_budget = options.thinking_budgets.get(options.thinking_level)
                if configured_budget is not None:
                    budget_tokens = int(configured_budget)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        kwargs = dict(await maybe_override_payload(kwargs, model, options))

        # ---------------------------------------------------------------
        # Track state across streaming events:
        # content_block_start tells us what type of block started (text / tool_use)
        # content_block_delta carries the actual payload
        # ---------------------------------------------------------------
        current_block_type: Optional[str] = None  # "text" | "tool_use" | "thinking"
        current_tool: Optional[Dict[str, Any]] = None  # accumulates tool id/name/json
        current_tool_json: str = ""

        async with client.messages.stream(**kwargs) as stream_resp:
            async for event in stream_resp:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    current_block_type = block.type
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and delta.text:
                        yield content_chunk(delta.text)
                    elif delta.type == "thinking_delta":
                        thinking_text = getattr(delta, "thinking", None) or getattr(delta, "text", None)
                        if thinking_text:
                            yield reasoning_chunk(str(thinking_text))
                    elif delta.type == "signature_delta":
                        signature = getattr(delta, "signature", None)
                        if signature:
                            yield provider_state_chunk("anthropic", {"thought_signatures": [str(signature)]})
                    elif delta.type == "input_json_delta" and delta.partial_json:
                        current_tool_json += delta.partial_json

                elif event_type == "content_block_stop":
                    if current_block_type == "tool_use" and current_tool:
                        # Emit the fully assembled tool call
                        args = parse_json_arguments(current_tool_json)
                        yield tool_calls_chunk(
                            [{
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "arguments": args,
                            }]
                        )
                        current_tool = None
                        current_tool_json = ""
                    current_block_type = None

                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    normalized_usage = _normalize_anthropic_usage(usage)
                    if normalized_usage:
                        yield usage_chunk(normalized_usage)


def _normalize_anthropic_usage(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None

    def read(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        return int(value) if value is not None else None

    extra: Dict[str, Any] = {}
    server_tool_use = read("server_tool_use")
    if server_tool_use is not None:
        extra["server_tool_use"] = server_tool_use

    return normalize_usage_payload(
        input_tokens=read("input_tokens"),
        output_tokens=read("output_tokens"),
        cache_read_tokens=read("cache_read_input_tokens"),
        cache_write_tokens=read("cache_creation_input_tokens"),
        extra=extra or None,
    )
