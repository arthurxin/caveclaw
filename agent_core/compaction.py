from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .types import (
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AssistantMessage,
    BaseContentBlock,
    ContentBlock,
    ImageBlock,
    Message,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultMessage,
)


def _is_message_excluded_from_llm(message: Message) -> bool:
    metadata = message.metadata
    return bool(metadata.get("exclude_from_llm") or metadata.get("ui_only"))


def _is_block_excluded_from_llm(block: BaseContentBlock) -> bool:
    metadata = block.metadata
    return bool(metadata.get("exclude_from_llm") or metadata.get("ui_only") or metadata.get("log_only"))


def _compact_runtime_entry_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    compacted: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compacted[key] = value
    return compacted


def _compact_runtime_snapshot_block(block: RuntimeSnapshotBlock) -> RuntimeSnapshotBlock:
    compacted_entries: List[RuntimeSnapshotEntry] = []
    for entry in block.entries:
        summary_blocks = compact_content_blocks(entry.summary_blocks)
        compacted_entries.append(
            RuntimeSnapshotEntry(
                key=entry.key,
                version=entry.version,
                summary_blocks=summary_blocks,
                metadata=_compact_runtime_entry_metadata(entry.metadata),
            )
        )
    return RuntimeSnapshotBlock(entries=compacted_entries, metadata=dict(block.metadata))


def compact_content_blocks(blocks: Iterable[ContentBlock]) -> List[ContentBlock]:
    compacted: List[ContentBlock] = []
    for block in blocks:
        if _is_block_excluded_from_llm(block):
            continue
        if isinstance(block, RuntimeSnapshotBlock):
            compacted.append(_compact_runtime_snapshot_block(block))
            continue
        if isinstance(block, ImageBlock):
            # Keep only lightweight metadata for provider-side serialization hooks.
            compacted.append(
                ImageBlock(
                    image_url=block.image_url,
                    mime_type=block.mime_type,
                    alt_text=block.alt_text,
                    metadata=dict(block.metadata),
                )
            )
            continue
        if isinstance(block, TextBlock):
            compacted.append(TextBlock(text=block.text, metadata=dict(block.metadata)))
            continue
        if isinstance(block, ThinkingBlock):
            compacted.append(
                ThinkingBlock(thinking=block.thinking, signature=block.signature, metadata=dict(block.metadata))
            )
            continue
        if isinstance(block, ToolCallBlock):
            compacted.append(
                ToolCallBlock(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.arguments),
                    metadata=dict(block.metadata),
                )
            )
            continue
        compacted.append(block)
    return compacted


def compact_message_for_llm(message: AgentMessage) -> Optional[AgentMessage]:
    if not isinstance(message, Message):
        return message
    if _is_message_excluded_from_llm(message):
        return None

    compacted_blocks = compact_content_blocks(message.content_blocks)
    if not compacted_blocks and message.raw_content is None and not message.metadata.get("allow_empty_for_llm"):
        return None

    base_kwargs = {
        "role": message.role,
        "content_blocks": compacted_blocks,
        "raw_content": message.raw_content,
        "provider_state": dict(message.provider_state) if message.provider_state else None,
        "metadata": dict(message.metadata),
        "timestamp": message.timestamp,
    }

    if isinstance(message, AssistantMessage):
        return AssistantMessage(
            **base_kwargs,
            tool_calls=[ToolCall(id=tool.id, name=tool.name, arguments=dict(tool.arguments)) for tool in message.tool_calls or []]
            or None,
            stop_reason=message.stop_reason,
            model=message.model,
            provider=message.provider,
            api=message.api,
            usage=dict(message.usage) if message.usage else None,
        )

    if isinstance(message, ToolResultMessage):
        return ToolResultMessage(
            **base_kwargs,
            tool_call_id=message.tool_call_id,
            name=message.name,
            details=message.details,
            is_error=message.is_error,
        )

    return Message(**base_kwargs)


async def compact_messages_for_llm(
    messages: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
) -> List[AgentMessage]:
    compacted_messages: List[AgentMessage] = []
    for message in messages:
        compacted_message = compact_message_for_llm(message)
        if compacted_message is not None:
            compacted_messages.append(compacted_message)

    custom_compactor = getattr(config, "compact_messages", None)
    if custom_compactor:
        compacted_messages = await custom_compactor(compacted_messages, context)

    return compacted_messages
