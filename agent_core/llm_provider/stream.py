from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, AsyncGenerator, List, Optional, TYPE_CHECKING

from .api_registry import SimpleStreamOptions, StreamOptions, api_provider_registry
from .env_api_keys import get_env_api_key
from .message_codec import codec_for_provider
from .provider_types import Model
from .usage import finalize_usage_payload

if TYPE_CHECKING:
    from ..assistant_messages.types import AssistantMessage, Message
    from .codecs.base import ProviderMessageCodec


@dataclass
class PreparedProviderStream:
    message_codec: "ProviderMessageCodec"
    options: StreamOptions
    partial_message: "AssistantMessage"
    chunk_stream: AsyncGenerator[dict[str, Any], None]


def _build_assistant_from_chunks(
    model: Model,
) -> "AssistantMessage":
    from ..assistant_messages.types import AssistantMessage

    return AssistantMessage(
        content_blocks=[],
        raw_content="",
        model=model.id,
        provider=model.provider,
        api=model.api,
    )


def _normalize_stream_options(options: Optional[StreamOptions | SimpleStreamOptions]) -> StreamOptions:
    if options is None:
        return StreamOptions()
    if isinstance(options, SimpleStreamOptions):
        option_payload = dict(options.__dict__)
        option_payload.pop("reasoning", None)
        normalized = StreamOptions(**option_payload)
        if options.reasoning is not None:
            normalized.thinking_level = options.reasoning
        return normalized
    return replace(options)


def prepare_provider_stream(
    model: Model,
    messages: List["Message"],
    *,
    options: Optional[StreamOptions | SimpleStreamOptions] = None,
) -> PreparedProviderStream:
    from ..assistant_messages.assistant_stream import append_assistant_delta
    from ..assistant_messages.transform_messages import (
        default_normalize_tool_call_id,
        should_normalize_tool_call_ids,
        transform_messages,
    )

    provider = api_provider_registry.get(model.api)
    if provider is None:
        raise ValueError(
            f"No ApiProvider registered for api type '{model.api}'. "
            f"Did you forget to call register_builtin_providers() or api_provider_registry.register(...)?"
        )

    effective_options = _normalize_stream_options(options)
    if effective_options.api_key is None:
        effective_options.api_key = get_env_api_key(model.provider)

    message_codec = codec_for_provider(provider)
    transformed_messages = transform_messages(
        messages,
        model,
        normalize_tool_call_id=default_normalize_tool_call_id if should_normalize_tool_call_ids(model) else None,
    )
    provider_messages = message_codec.encode_messages(transformed_messages, effective_options)

    partial_message = _build_assistant_from_chunks(model)
    
    async def _normalized_chunks() -> AsyncGenerator[dict[str, Any], None]:
        async for raw_chunk in provider.stream(model, provider_messages, effective_options, api_key=effective_options.api_key):
            normalized_chunk = message_codec.decode_chunk(raw_chunk, partial_message)
            append_assistant_delta(partial_message, normalized_chunk)
            yield normalized_chunk

    return PreparedProviderStream(
        message_codec=message_codec,
        options=effective_options,
        partial_message=partial_message,
        chunk_stream=_normalized_chunks(),
    )


async def stream(
    model: Model,
    messages: List["Message"],
    *,
    options: Optional[StreamOptions | SimpleStreamOptions] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    prepared = prepare_provider_stream(model, messages, options=options)
    async for chunk in prepared.chunk_stream:
        yield chunk


async def complete(
    model: Model,
    messages: List["Message"],
    *,
    options: Optional[StreamOptions | SimpleStreamOptions] = None,
) -> "AssistantMessage":
    from ..assistant_messages.assistant_stream import append_assistant_delta

    prepared = prepare_provider_stream(model, messages, options=options)
    assistant_message = _build_assistant_from_chunks(model)

    async for raw_chunk in prepared.chunk_stream:
        append_assistant_delta(assistant_message, raw_chunk)

    if assistant_message.raw_content == "":
        assistant_message.raw_content = None
    if assistant_message.stop_reason is None:
        assistant_message.stop_reason = "tool_use" if assistant_message.tool_calls else "stop"
    prepared.message_codec.finalize_provider_state(assistant_message)
    prepared.message_codec.finalize_assistant_message(assistant_message)
    if assistant_message.usage is not None:
        assistant_message.usage = finalize_usage_payload(assistant_message.usage, model)
    return assistant_message


async def stream_simple(
    model: Model,
    messages: List["Message"],
    *,
    options: Optional[SimpleStreamOptions] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    async for chunk in stream(model, messages, options=options):
        yield chunk


async def complete_simple(
    model: Model,
    messages: List["Message"],
    *,
    options: Optional[SimpleStreamOptions] = None,
) -> "AssistantMessage":
    return await complete(model, messages, options=options)


__all__ = ["PreparedProviderStream", "complete", "complete_simple", "prepare_provider_stream", "stream", "stream_simple"]
