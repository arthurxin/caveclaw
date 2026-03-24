from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

from ..llm_provider.provider_types import Model
from .content_blocks import (
    BaseContentBlock,
    ContentBlock,
    ImageBlock,
    RuntimeRefBlock,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultBlock,
    content_blocks_to_text,
)
from .messages import AssistantMessage, Message, ToolResultMessage


NormalizeToolCallId = Callable[[str, Model, AssistantMessage], str]
ProviderStatePolicy = Literal["drop", "same-model-only", "preserve"]
RuntimeSnapshotPolicy = Literal["drop", "render-text", "same-model-only", "preserve"]
SyntheticToolResultPolicy = Literal["insert", "skip"]
AssistantFailurePolicy = Literal["skip", "keep"]

_INVALID_TOOL_CALL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class TransformMessagesOptions:
    provider_state_policy: ProviderStatePolicy = "same-model-only"
    runtime_snapshot_policy: RuntimeSnapshotPolicy = "same-model-only"
    synthetic_tool_result_policy: SyntheticToolResultPolicy = "insert"
    assistant_failure_policy: AssistantFailurePolicy = "skip"
    normalize_tool_call_id: Optional[NormalizeToolCallId] = None


@dataclass
class ToolCallIdRewrite:
    message_index: int
    tool_name: str
    source_id: str
    target_id: str


@dataclass
class SyntheticToolResultDetail:
    message_index: int
    tool_call_id: str
    tool_name: str


@dataclass
class SkippedAssistantDetail:
    message_index: int
    stop_reason: Optional[str]


@dataclass
class ProviderStateDropDetail:
    message_index: int
    role: str
    namespaces: List[str]
    reason: str


@dataclass
class RuntimeBlockConversionDetail:
    message_index: int
    role: str
    block_type: str
    mode: str


@dataclass
class TransformMessagesResult:
    messages: List[Message]
    tool_call_id_map: Dict[str, str] = field(default_factory=dict)
    synthetic_tool_results_added: int = 0
    skipped_assistant_messages: int = 0
    provider_state_drops: int = 0
    runtime_snapshot_conversions: int = 0
    tool_call_id_rewrites: List[ToolCallIdRewrite] = field(default_factory=list)
    synthetic_tool_result_details: List[SyntheticToolResultDetail] = field(default_factory=list)
    skipped_assistant_details: List[SkippedAssistantDetail] = field(default_factory=list)
    provider_state_drop_details: List[ProviderStateDropDetail] = field(default_factory=list)
    runtime_block_conversion_details: List[RuntimeBlockConversionDetail] = field(default_factory=list)


def default_normalize_tool_call_id(tool_call_id: str, model: Model, source: AssistantMessage) -> str:
    sanitized = _INVALID_TOOL_CALL_ID_CHARS.sub("_", tool_call_id).strip("_")
    if not sanitized:
        sanitized = "tool_call"

    if len(sanitized) <= 64:
        return sanitized

    digest = hashlib.sha1(tool_call_id.encode("utf-8")).hexdigest()[:12]
    prefix = sanitized[: max(1, 64 - len(digest) - 1)]
    return f"{prefix}_{digest}"


def should_normalize_tool_call_ids(model: Model) -> bool:
    compat = getattr(model, "compat", None)
    if compat and getattr(compat, "requiresMistralToolIds", None):
        return True
    api = getattr(model, "api", "") or ""
    provider = getattr(model, "provider", "") or ""
    return api == "anthropic-messages" or provider in {"anthropic", "mistral"}


def transform_messages(
    messages: List[Message],
    model: Model,
    *,
    normalize_tool_call_id: Optional[NormalizeToolCallId] = None,
) -> List[Message]:
    options = TransformMessagesOptions(normalize_tool_call_id=normalize_tool_call_id)
    return transform_messages_with_result(messages, model, options=options).messages


def transform_messages_with_result(
    messages: List[Message],
    model: Model,
    *,
    options: Optional[TransformMessagesOptions] = None,
) -> TransformMessagesResult:
    effective_options = _resolve_transform_options(model, options)
    result = TransformMessagesResult(messages=[])

    transformed = [
        _transform_single_message(
            message,
            model,
            message_index=index,
            options=effective_options,
            result=result,
        )
        for index, message in enumerate(messages)
    ]
    result.messages = _insert_synthetic_tool_results(
        transformed,
        result=result,
        options=effective_options,
    )
    return result


def _resolve_transform_options(
    model: Model,
    options: Optional[TransformMessagesOptions],
) -> TransformMessagesOptions:
    effective_options = options or TransformMessagesOptions()
    normalizer = effective_options.normalize_tool_call_id
    if normalizer is None and should_normalize_tool_call_ids(model):
        normalizer = default_normalize_tool_call_id
    return TransformMessagesOptions(
        provider_state_policy=effective_options.provider_state_policy,
        runtime_snapshot_policy=effective_options.runtime_snapshot_policy,
        synthetic_tool_result_policy=effective_options.synthetic_tool_result_policy,
        assistant_failure_policy=effective_options.assistant_failure_policy,
        normalize_tool_call_id=normalizer,
    )


def _transform_single_message(
    message: Message,
    model: Model,
    *,
    message_index: int,
    options: TransformMessagesOptions,
    result: TransformMessagesResult,
) -> Message:
    tool_call_id_map = result.tool_call_id_map

    if isinstance(message, ToolResultMessage):
        normalized_id = tool_call_id_map.get(message.tool_call_id, message.tool_call_id)
        transformed_provider_state = _transform_provider_state(
            message.provider_state,
            message_index=message_index,
            role=message.role,
            is_same_model=False,
            options=options,
            result=result,
        )
        return ToolResultMessage(
            role=message.role,
            tool_call_id=normalized_id,
            name=message.name,
            content_blocks=_transform_content_blocks(
                message.content_blocks,
                is_same_model=False,
                preserve_structured_state=False,
                model=model,
                message_index=message_index,
                role=message.role,
                options=options,
                source_assistant=None,
                result=result,
            ),
            raw_content=message.raw_content,
            details=message.details,
            is_error=message.is_error,
            provider_state=transformed_provider_state,
            metadata=dict(message.metadata),
            timestamp=message.timestamp,
        )

    if isinstance(message, AssistantMessage):
        is_same_model = _is_same_model_message(message, model)
        preserve_structured_state = is_same_model and options.provider_state_policy != "drop"
        return AssistantMessage(
            role=message.role,
            content_blocks=_transform_content_blocks(
                message.content_blocks,
                is_same_model=is_same_model,
                preserve_structured_state=preserve_structured_state,
                model=model,
                message_index=message_index,
                role=message.role,
                options=options,
                source_assistant=message,
                result=result,
            ),
            raw_content=message.raw_content if preserve_structured_state else None,
            tool_calls=_transform_tool_calls(
                message.tool_calls,
                message_index=message_index,
                is_same_model=is_same_model,
                model=model,
                source_assistant=message,
                options=options,
                result=result,
            ),
            stop_reason=message.stop_reason,
            model=message.model,
            provider=message.provider,
            api=message.api,
            usage=dict(message.usage) if message.usage else None,
            provider_state=_transform_provider_state(
                message.provider_state,
                message_index=message_index,
                role=message.role,
                is_same_model=is_same_model,
                options=options,
                result=result,
            ),
            metadata=dict(message.metadata),
            timestamp=message.timestamp,
        )

    return Message(
        role=message.role,
        content_blocks=_transform_content_blocks(
            message.content_blocks,
            is_same_model=False,
            preserve_structured_state=False,
            model=model,
            message_index=message_index,
            role=message.role,
            options=options,
            source_assistant=None,
            result=result,
        ),
        raw_content=message.raw_content,
        provider_state=_transform_provider_state(
            message.provider_state,
            message_index=message_index,
            role=message.role,
            is_same_model=False,
            options=options,
            result=result,
        ),
        metadata=dict(message.metadata),
        timestamp=message.timestamp,
    )


def _transform_provider_state(
    provider_state: Optional[Dict[str, object]],
    *,
    message_index: int,
    role: str,
    is_same_model: bool,
    options: TransformMessagesOptions,
    result: TransformMessagesResult,
) -> Optional[Dict[str, object]]:
    if not provider_state:
        return None
    if options.provider_state_policy == "preserve":
        return dict(provider_state)
    if options.provider_state_policy == "same-model-only" and is_same_model:
        return dict(provider_state)
    result.provider_state_drops += 1
    result.provider_state_drop_details.append(
        ProviderStateDropDetail(
            message_index=message_index,
            role=role,
            namespaces=sorted(str(namespace) for namespace in provider_state.keys()),
            reason=options.provider_state_policy,
        )
    )
    return None


def _transform_tool_calls(
    tool_calls: Optional[List[ToolCall]],
    *,
    message_index: int,
    is_same_model: bool,
    model: Model,
    source_assistant: AssistantMessage,
    options: TransformMessagesOptions,
    result: TransformMessagesResult,
) -> Optional[List[ToolCall]]:
    if not tool_calls:
        return None

    transformed: List[ToolCall] = []
    for tool_call in tool_calls:
        normalized_id = _normalize_tool_call_id(
            tool_call.id,
            message_index=message_index,
            tool_name=tool_call.name,
            is_same_model=is_same_model,
            model=model,
            source_assistant=source_assistant,
            options=options,
            result=result,
        )
        transformed.append(
            ToolCall(
                id=normalized_id,
                name=tool_call.name,
                arguments=dict(tool_call.arguments),
            )
        )
    return transformed


def _transform_content_blocks(
    blocks: List[ContentBlock],
    *,
    is_same_model: bool,
    preserve_structured_state: bool,
    model: Model,
    message_index: int,
    role: str,
    options: TransformMessagesOptions,
    source_assistant: Optional[AssistantMessage],
    result: TransformMessagesResult,
) -> List[ContentBlock]:
    transformed: List[ContentBlock] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            transformed.append(TextBlock(text=block.text, metadata=dict(block.metadata)))
            continue

        if isinstance(block, ImageBlock):
            transformed.append(
                ImageBlock(
                    image_url=block.image_url,
                    mime_type=block.mime_type,
                    alt_text=block.alt_text,
                    metadata=dict(block.metadata),
                )
            )
            continue

        if isinstance(block, ThinkingBlock):
            if preserve_structured_state:
                transformed.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=block.signature,
                        metadata=dict(block.metadata),
                    )
                )
            elif block.thinking.strip():
                transformed.append(TextBlock(text=block.thinking, metadata=dict(block.metadata)))
            continue

        if isinstance(block, ToolCallBlock):
            normalized_id = _normalize_tool_call_id(
                block.id,
                message_index=message_index,
                tool_name=block.name,
                is_same_model=is_same_model,
                model=model,
                source_assistant=source_assistant,
                options=options,
                result=result,
            )
            transformed.append(
                ToolCallBlock(
                    id=normalized_id,
                    name=block.name,
                    arguments=dict(block.arguments),
                    metadata=dict(block.metadata),
                )
            )
            continue

        if isinstance(block, ToolResultBlock):
            transformed.append(
                ToolResultBlock(
                    tool_call_id=result.tool_call_id_map.get(block.tool_call_id, block.tool_call_id),
                    tool_name=block.tool_name,
                    content_blocks=_transform_content_blocks(
                        block.content_blocks,
                        is_same_model=is_same_model,
                        preserve_structured_state=preserve_structured_state,
                        model=model,
                        message_index=message_index,
                        role=role,
                        options=options,
                        source_assistant=source_assistant,
                        result=result,
                    ),
                    is_error=block.is_error,
                    details=block.details,
                    metadata=dict(block.metadata),
                )
            )
            continue

        if isinstance(block, RuntimeRefBlock):
            if _should_preserve_runtime_block(preserve_structured_state=preserve_structured_state, options=options):
                transformed.append(
                    RuntimeRefBlock(
                        key=block.key,
                        version=block.version,
                        label=block.label,
                        metadata=dict(block.metadata),
                    )
                )
            elif options.runtime_snapshot_policy != "drop":
                transformed.append(TextBlock(text=_render_runtime_ref(block), metadata=dict(block.metadata)))
                result.runtime_snapshot_conversions += 1
                result.runtime_block_conversion_details.append(
                    RuntimeBlockConversionDetail(
                        message_index=message_index,
                        role=role,
                        block_type="runtime_ref",
                        mode="render-text",
                    )
                )
            continue

        if isinstance(block, RuntimeSnapshotBlock):
            if _should_preserve_runtime_block(preserve_structured_state=preserve_structured_state, options=options):
                transformed.append(_clone_runtime_snapshot_block(block))
            elif options.runtime_snapshot_policy == "render-text":
                transformed.append(TextBlock(text=_render_runtime_snapshot(block), metadata=dict(block.metadata)))
                result.runtime_snapshot_conversions += 1
                result.runtime_block_conversion_details.append(
                    RuntimeBlockConversionDetail(
                        message_index=message_index,
                        role=role,
                        block_type="runtime_snapshot",
                        mode="render-text",
                    )
                )
            elif options.runtime_snapshot_policy == "same-model-only" and not preserve_structured_state:
                transformed.append(TextBlock(text=_render_runtime_snapshot(block), metadata=dict(block.metadata)))
                result.runtime_snapshot_conversions += 1
                result.runtime_block_conversion_details.append(
                    RuntimeBlockConversionDetail(
                        message_index=message_index,
                        role=role,
                        block_type="runtime_snapshot",
                        mode="same-model-only",
                    )
                )
            continue

        transformed.append(_clone_base_block(block))

    return transformed


def _should_preserve_runtime_block(*, preserve_structured_state: bool, options: TransformMessagesOptions) -> bool:
    if options.runtime_snapshot_policy == "preserve":
        return True
    if options.runtime_snapshot_policy == "same-model-only":
        return preserve_structured_state
    return False


def _normalize_tool_call_id(
    tool_call_id: str,
    *,
    message_index: int,
    tool_name: str,
    is_same_model: bool,
    model: Model,
    source_assistant: Optional[AssistantMessage],
    options: TransformMessagesOptions,
    result: TransformMessagesResult,
) -> str:
    if tool_call_id in result.tool_call_id_map:
        return result.tool_call_id_map[tool_call_id]

    normalized_id = tool_call_id
    if not is_same_model and options.normalize_tool_call_id is not None and source_assistant is not None:
        normalized_id = options.normalize_tool_call_id(tool_call_id, model, source_assistant)

    if normalized_id != tool_call_id:
        result.tool_call_id_map[tool_call_id] = normalized_id
        result.tool_call_id_rewrites.append(
            ToolCallIdRewrite(
                message_index=message_index,
                tool_name=tool_name,
                source_id=tool_call_id,
                target_id=normalized_id,
            )
        )
    return normalized_id


def _is_same_model_message(message: AssistantMessage, model: Model) -> bool:
    return (
        getattr(message, "provider", None) == getattr(model, "provider", None)
        and getattr(message, "api", None) == getattr(model, "api", None)
        and getattr(message, "model", None) == getattr(model, "id", None)
    )


def _assistant_tool_calls(message: AssistantMessage) -> List[ToolCall]:
    if message.tool_calls:
        return [ToolCall(id=tool_call.id, name=tool_call.name, arguments=dict(tool_call.arguments)) for tool_call in message.tool_calls]

    tool_calls: List[ToolCall] = []
    for block in message.content_blocks:
        if isinstance(block, ToolCallBlock):
            tool_calls.append(block.to_tool_call())
    return tool_calls


def _insert_synthetic_tool_results(
    messages: List[Message],
    *,
    result: TransformMessagesResult,
    options: TransformMessagesOptions,
) -> List[Message]:
    output: List[Message] = []
    pending_tool_calls: List[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    for message in messages:
        if isinstance(message, AssistantMessage):
            output.extend(
                _build_missing_tool_results(
                    pending_tool_calls,
                    existing_tool_result_ids,
                    message_index=len(output),
                    result=result,
                    options=options,
                )
            )
            pending_tool_calls = []
            existing_tool_result_ids = set()

            if message.stop_reason in {"error", "aborted"} and options.assistant_failure_policy == "skip":
                result.skipped_assistant_messages += 1
                result.skipped_assistant_details.append(
                    SkippedAssistantDetail(
                        message_index=len(output),
                        stop_reason=message.stop_reason,
                    )
                )
                continue

            pending_tool_calls = _assistant_tool_calls(message)
            output.append(message)
            continue

        if isinstance(message, ToolResultMessage):
            existing_tool_result_ids.add(message.tool_call_id)
            output.append(message)
            continue

        output.extend(
            _build_missing_tool_results(
                pending_tool_calls,
                existing_tool_result_ids,
                message_index=len(output),
                result=result,
                options=options,
            )
        )
        pending_tool_calls = []
        existing_tool_result_ids = set()
        output.append(message)

    return output


def _build_missing_tool_results(
    pending_tool_calls: List[ToolCall],
    existing_tool_result_ids: set[str],
    *,
    message_index: int,
    result: TransformMessagesResult,
    options: TransformMessagesOptions,
) -> List[ToolResultMessage]:
    if options.synthetic_tool_result_policy == "skip":
        return []

    synthetic_results: List[ToolResultMessage] = []
    for tool_call in pending_tool_calls:
        if tool_call.id in existing_tool_result_ids:
            continue
        synthetic_results.append(
            ToolResultMessage(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="No result provided",
                is_error=True,
            )
        )
        result.synthetic_tool_results_added += 1
        result.synthetic_tool_result_details.append(
            SyntheticToolResultDetail(
                message_index=message_index,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            )
        )
    return synthetic_results


def _render_runtime_snapshot(block: RuntimeSnapshotBlock) -> str:
    lines = ["Runtime Snapshot:"]
    for entry in block.entries:
        summary = _render_content_blocks_as_text(entry.summary_blocks)
        header = f"- {entry.key}@v{entry.version}"
        metadata = _render_runtime_metadata(entry.metadata)
        if metadata:
            header = f"{header} [{metadata}]"
        if summary:
            lines.append(f"{header}: {summary}")
        else:
            lines.append(header)
    return "\n".join(lines)


def _render_runtime_ref(block: RuntimeRefBlock) -> str:
    label = block.label or block.key
    if block.version is None:
        return f"Runtime Ref: {label} ({block.key})"
    return f"Runtime Ref: {label} ({block.key}@v{block.version})"


def _render_runtime_metadata(metadata: Dict[str, object]) -> str:
    visible = [f"{key}={value}" for key, value in sorted(metadata.items()) if isinstance(value, (str, int, float, bool)) or value is None]
    return ", ".join(visible)


def _render_content_blocks_as_text(blocks: List[ContentBlock]) -> str:
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            if block.text:
                parts.append(block.text)
            continue
        if isinstance(block, ThinkingBlock):
            if block.thinking.strip():
                parts.append(block.thinking)
            continue
        if isinstance(block, RuntimeRefBlock):
            parts.append(_render_runtime_ref(block))
            continue
        if isinstance(block, RuntimeSnapshotBlock):
            parts.append(_render_runtime_snapshot(block))
            continue
        if isinstance(block, ToolResultBlock):
            nested = _render_content_blocks_as_text(block.content_blocks)
            if nested:
                parts.append(nested)
            continue
    if parts:
        return "\n".join(part for part in parts if part)
    return content_blocks_to_text(blocks)


def _clone_runtime_snapshot_block(block: RuntimeSnapshotBlock) -> RuntimeSnapshotBlock:
    return RuntimeSnapshotBlock(
        entries=[
            RuntimeSnapshotEntry(
                key=entry.key,
                version=entry.version,
                summary_blocks=_clone_content_blocks(entry.summary_blocks),
                metadata=dict(entry.metadata),
            )
            for entry in block.entries
        ],
        metadata=dict(block.metadata),
    )


def _clone_content_blocks(blocks: List[ContentBlock]) -> List[ContentBlock]:
    cloned: List[ContentBlock] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            cloned.append(TextBlock(text=block.text, metadata=dict(block.metadata)))
        elif isinstance(block, ThinkingBlock):
            cloned.append(ThinkingBlock(thinking=block.thinking, signature=block.signature, metadata=dict(block.metadata)))
        elif isinstance(block, RuntimeRefBlock):
            cloned.append(RuntimeRefBlock(key=block.key, version=block.version, label=block.label, metadata=dict(block.metadata)))
        elif isinstance(block, RuntimeSnapshotBlock):
            cloned.append(_clone_runtime_snapshot_block(block))
        elif isinstance(block, ToolCallBlock):
            cloned.append(ToolCallBlock(id=block.id, name=block.name, arguments=dict(block.arguments), metadata=dict(block.metadata)))
        elif isinstance(block, ToolResultBlock):
            cloned.append(
                ToolResultBlock(
                    tool_call_id=block.tool_call_id,
                    tool_name=block.tool_name,
                    content_blocks=_clone_content_blocks(block.content_blocks),
                    is_error=block.is_error,
                    details=block.details,
                    metadata=dict(block.metadata),
                )
            )
        elif isinstance(block, ImageBlock):
            cloned.append(
                ImageBlock(
                    image_url=block.image_url,
                    mime_type=block.mime_type,
                    alt_text=block.alt_text,
                    metadata=dict(block.metadata),
                )
            )
        else:
            cloned.append(_clone_base_block(block))
    return cloned


def _clone_base_block(block: BaseContentBlock) -> BaseContentBlock:
    return BaseContentBlock(type=block.type, metadata=dict(block.metadata))


__all__ = [
    "AssistantFailurePolicy",
    "NormalizeToolCallId",
    "ProviderStateDropDetail",
    "ProviderStatePolicy",
    "RuntimeBlockConversionDetail",
    "RuntimeSnapshotPolicy",
    "SkippedAssistantDetail",
    "SyntheticToolResultDetail",
    "SyntheticToolResultPolicy",
    "ToolCallIdRewrite",
    "TransformMessagesOptions",
    "TransformMessagesResult",
    "default_normalize_tool_call_id",
    "should_normalize_tool_call_ids",
    "transform_messages",
    "transform_messages_with_result",
]
