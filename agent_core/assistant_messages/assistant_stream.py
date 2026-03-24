from __future__ import annotations

import uuid
import inspect
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .types import (
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AssistantDelta,
    AssistantMessage,
    BasicCancellationSignal,
    CancellationSignal,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
)

def _merge_usage(existing: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
    from ..llm_provider.usage import finalize_usage_payload

    merged = dict(existing) if existing else {}
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return finalize_usage_payload(merged)


def _normalize_assistant_delta(delta: Any) -> AssistantDelta:
    if isinstance(delta, AssistantDelta):
        return delta
    if isinstance(delta, dict):
        return AssistantDelta.from_chunk(delta)
    raise TypeError(f"Unsupported assistant delta payload: {type(delta)!r}")


def append_assistant_delta(assistant_message: AssistantMessage, chunk: Dict[str, Any] | AssistantDelta) -> None:
    normalized = _normalize_assistant_delta(chunk)
    chunk_payload = normalized.to_chunk()

    raw_content_override = chunk_payload.get("raw_content")
    if raw_content_override is not None:
        assistant_message.raw_content = str(raw_content_override)

    if "content" in chunk_payload and chunk_payload["content"]:
        delta_text = str(chunk_payload["content"])
        if assistant_message.content_blocks and isinstance(assistant_message.content_blocks[-1], TextBlock):
            assistant_message.content_blocks[-1].text += delta_text
        else:
            assistant_message.content_blocks.append(TextBlock(text=delta_text))
        if raw_content_override is None:
            if assistant_message.raw_content is None:
                assistant_message.raw_content = ""
            assistant_message.raw_content += delta_text

    if "reasoning" in chunk_payload and chunk_payload["reasoning"]:
        delta_reasoning = str(chunk_payload["reasoning"])
        if assistant_message.content_blocks and isinstance(assistant_message.content_blocks[-1], ThinkingBlock):
            assistant_message.content_blocks[-1].thinking += delta_reasoning
        else:
            assistant_message.content_blocks.append(ThinkingBlock(thinking=delta_reasoning))
        if assistant_message.raw_content is not None and raw_content_override is None:
            assistant_message.raw_content += f"\n<think>{delta_reasoning}</think>"

    if "tool_calls" in chunk_payload and chunk_payload["tool_calls"]:
        if not assistant_message.tool_calls:
            assistant_message.tool_calls = []
        for tool_call_payload in chunk_payload["tool_calls"]:
            tool_call = ToolCall(**tool_call_payload)
            assistant_message.tool_calls.append(tool_call)
            assistant_message.content_blocks.append(
                ToolCallBlock(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)
            )

    if "provider_state" in chunk_payload and isinstance(chunk_payload["provider_state"], dict):
        if assistant_message.provider_state is None:
            assistant_message.provider_state = {}
        for namespace, payload in chunk_payload["provider_state"].items():
            if isinstance(payload, dict):
                existing_payload = assistant_message.provider_state.get(namespace)
                if isinstance(existing_payload, dict):
                    merged_payload = dict(existing_payload)
                    for key, value in payload.items():
                        if isinstance(value, list) and isinstance(merged_payload.get(key), list):
                            merged_payload[key] = list(merged_payload[key]) + list(value)
                        else:
                            merged_payload[key] = value
                    assistant_message.provider_state[namespace] = merged_payload
                else:
                    assistant_message.provider_state[namespace] = dict(payload)

    if "usage" in chunk_payload and isinstance(chunk_payload["usage"], dict):
        assistant_message.usage = _merge_usage(assistant_message.usage, chunk_payload["usage"])


async def stream_assistant_response(
    messages: List[Message],
    tools: List[AgentTool],
    config: AgentLoopConfig,
    stream_fn: Optional[Callable[..., AsyncGenerator[Any, None]]] = None,
    yield_event: Optional[Callable[[AgentEvent], None]] = None,
    message_parent_id: Optional[str] = None,
    signal: Optional[CancellationSignal] = None,
) -> AssistantMessage:
    assistant_message = AssistantMessage(content_blocks=[], raw_content="")
    message_event_id = str(uuid.uuid4())
    active_signal = signal or BasicCancellationSignal()
    message_codec = None

    if yield_event:
        yield_event(
            AgentEvent(
                type="message_start",
                event_id=message_event_id,
                parent_id=message_parent_id,
                data={"message": assistant_message},
            )
        )

    async def consume_chunks(chunk_stream: AsyncGenerator[Dict[str, Any], None]) -> None:
        async for chunk in chunk_stream:
            if active_signal.is_cancelled:
                assistant_message.stop_reason = "aborted"
                break
            normalized_chunk = (
                message_codec.decode_chunk(chunk, assistant_message)
                if message_codec is not None
                else chunk
            )
            assistant_delta = _normalize_assistant_delta(normalized_chunk)
            append_assistant_delta(assistant_message, assistant_delta)
            if yield_event:
                yield_event(
                    AgentEvent(
                        type="message_update",
                        parent_id=message_event_id,
                        data={"message": assistant_message, "delta": assistant_delta},
                    )
                )

    async def resolve_api_key(provider_name: str) -> Optional[str]:
        resolver = getattr(config, "get_api_key", None)
        if resolver is not None:
            resolved = resolver(provider_name)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved:
                return str(resolved)
        configured_api_key = getattr(config, "api_key", None)
        if configured_api_key:
            return str(configured_api_key)
        return None

    if stream_fn is not None:
        await consume_chunks(stream_fn(messages))
    else:
        from ..llm_provider.api_registry import StreamOptions
        from ..llm_provider.stream import prepare_provider_stream

        model = getattr(config, "model", None)
        if model is None:
            raise ValueError(
                "Neither `stream_fn` nor `config.model` was provided. "
                "Please set config.model to an agent_core.llm_provider.Model instance, "
                "or pass a `stream_fn` for legacy use."
            )

        assistant_message.model = getattr(model, "id", None)
        assistant_message.provider = getattr(model, "provider", None)
        assistant_message.api = getattr(model, "api", None)

        options = StreamOptions()
        options.thinking_level = getattr(config, "thinking_level", "off") or "off"
        options.system_prompt = getattr(config, "system_prompt", None)
        options.temperature = getattr(config, "temperature", None)
        options.max_tokens = getattr(config, "max_tokens", None)
        options.signal = active_signal
        options.tools = tools
        options.transport = getattr(config, "transport", None)
        options.cache_retention = getattr(config, "cache_retention", None)
        options.session_id = getattr(config, "session_id", None)
        options.on_payload = getattr(config, "on_payload", None)
        options.headers = getattr(config, "headers", None)
        options.max_retry_delay_ms = getattr(config, "max_retry_delay_ms", None)
        options.metadata = getattr(config, "metadata", None)
        options.thinking_budgets = getattr(config, "thinking_budgets", None)

        api_key = await resolve_api_key(model.provider)
        options.api_key = api_key

        prepared = prepare_provider_stream(
            model,
            messages,
            options=options,
        )
        message_codec = prepared.message_codec
        await consume_chunks(prepared.chunk_stream)

    if assistant_message.raw_content == "":
        assistant_message.raw_content = None

    if assistant_message.stop_reason is None:
        assistant_message.stop_reason = "tool_use" if assistant_message.tool_calls else "stop"

    if message_codec is not None:
        message_codec.finalize_provider_state(assistant_message)
        message_codec.finalize_assistant_message(assistant_message)
    model_for_usage = getattr(config, "model", None)
    if assistant_message.usage is not None:
        from ..llm_provider.usage import finalize_usage_payload

        assistant_message.usage = finalize_usage_payload(assistant_message.usage, model_for_usage)

    if yield_event:
        yield_event(
            AgentEvent(
                type="message_end",
                parent_id=message_event_id,
                data={"message": assistant_message},
            )
        )

    return assistant_message
