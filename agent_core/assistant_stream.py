from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .types import (
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AssistantMessage,
    BasicCancellationSignal,
    CancellationSignal,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
)


def append_assistant_delta(assistant_message: AssistantMessage, chunk: Dict[str, Any]) -> None:
    raw_content_override = chunk.get("raw_content")
    if raw_content_override is not None:
        assistant_message.raw_content = str(raw_content_override)

    if "content" in chunk and chunk["content"]:
        delta_text = str(chunk["content"])
        assistant_message.content_blocks.append(TextBlock(text=delta_text))
        if assistant_message.raw_content is None:
            assistant_message.raw_content = ""
            assistant_message.raw_content += delta_text

    if "reasoning" in chunk and chunk["reasoning"]:
        assistant_message.content_blocks.append(ThinkingBlock(thinking=str(chunk["reasoning"])))
        if assistant_message.raw_content is not None and raw_content_override is None:
            assistant_message.raw_content += f"\n<think>{chunk['reasoning']}</think>"

    if "tool_calls" in chunk and chunk["tool_calls"]:
        if not assistant_message.tool_calls:
            assistant_message.tool_calls = []
        for tool_call_payload in chunk["tool_calls"]:
            tool_call = ToolCall(**tool_call_payload)
            assistant_message.tool_calls.append(tool_call)
            assistant_message.content_blocks.append(
                ToolCallBlock(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)
            )

    if "provider_state" in chunk and isinstance(chunk["provider_state"], dict):
        if assistant_message.provider_state is None:
            assistant_message.provider_state = {}
        for namespace, payload in chunk["provider_state"].items():
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
            append_assistant_delta(assistant_message, chunk)
            if yield_event:
                yield_event(
                    AgentEvent(
                        type="message_update",
                        parent_id=message_event_id,
                        data={"message": assistant_message, "delta": chunk},
                    )
                )

    if stream_fn is not None:
        await consume_chunks(stream_fn(messages))
    else:
        from .llm.api_registry import StreamOptions, api_provider_registry
        from .llm.message_codec import codec_for_provider

        model = getattr(config, "model", None)
        if model is None:
            raise ValueError(
                "Neither `stream_fn` nor `config.model` was provided. "
                "Please set config.model to an agent_core.llm.Model instance, "
                "or pass a `stream_fn` for legacy use."
            )

        provider = api_provider_registry.get(model.api)
        if provider is None:
            raise ValueError(
                f"No ApiProvider registered for api type '{model.api}'. "
                f"Did you forget to call api_provider_registry.register(YourProvider())?"
            )

        options = StreamOptions()
        options.thinking_level = getattr(config, "thinking_level", "off") or "off"
        options.system_prompt = getattr(config, "system_prompt", None)
        options.tools = tools

        api_key = None
        registry = getattr(config, "model_registry", None)
        if registry and hasattr(registry, "get_api_key"):
            api_key = registry.get_api_key(model.provider)

        message_codec = codec_for_provider(provider)
        provider_messages = message_codec.to_provider_messages(messages, options)
        await consume_chunks(provider.stream(model, provider_messages, options, api_key=api_key))

    if assistant_message.raw_content == "":
        assistant_message.raw_content = None

    if assistant_message.stop_reason is None:
        assistant_message.stop_reason = "tool_use" if assistant_message.tool_calls else "stop"

    if yield_event:
        yield_event(
            AgentEvent(
                type="message_end",
                parent_id=message_event_id,
                data={"message": assistant_message},
            )
        )

    return assistant_message
