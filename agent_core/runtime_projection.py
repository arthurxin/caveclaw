from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, List, Optional

from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    ContentBlock,
    Message,
    RuntimeDeltaOp,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    RuntimeVariable,
    TextBlock,
    ToolCall,
    content_blocks_from_text,
)


def normalize_inspector_output(value: Any) -> RuntimeSnapshotBlock:
    if isinstance(value, RuntimeSnapshotBlock):
        return value
    if isinstance(value, list):
        return RuntimeSnapshotBlock(entries=[RuntimeSnapshotEntry(key="runtime", version=0, summary_blocks=value)])
    return RuntimeSnapshotBlock(
        entries=[
            RuntimeSnapshotEntry(
                key="runtime",
                version=0,
                summary_blocks=content_blocks_from_text(str(value)),
            )
        ]
    )


def build_variable_metadata_blocks(variable: RuntimeVariable) -> List[ContentBlock]:
    if variable.llm_view:
        return list(variable.llm_view)

    metadata_parts = [
        f"key={variable.key}",
        f"kind={variable.kind}",
        f"version={variable.version}",
    ]
    if variable.updated_by:
        metadata_parts.append(f"updated_by={variable.updated_by}")
    if variable.metadata:
        visible_metadata = ", ".join(f"{key}={value}" for key, value in sorted(variable.metadata.items()))
        metadata_parts.append(f"metadata={{ {visible_metadata} }}")
    return [TextBlock(text="RuntimeVariable(" + ", ".join(metadata_parts) + ")")]


def build_runtime_snapshot(context: AgentContext, config: AgentLoopConfig) -> Optional[RuntimeSnapshotBlock]:
    inspector = getattr(config, "inspector", None)
    if inspector:
        return None

    if not context.runtime.variables:
        return None

    entries: List[RuntimeSnapshotEntry] = []
    for key, variable in context.runtime.variables.items():
        entries.append(
            RuntimeSnapshotEntry(
                key=key,
                version=variable.version,
                summary_blocks=build_variable_metadata_blocks(variable),
                metadata=dict(variable.metadata),
            )
        )
    return RuntimeSnapshotBlock(entries=entries)


async def inject_runtime_snapshot(
    context_messages: List[AgentMessage],
    agent_context: AgentContext,
    config: AgentLoopConfig,
) -> Optional[Message]:
    snapshot_block: Optional[RuntimeSnapshotBlock] = None
    inspector = getattr(config, "inspector", None)

    current_runtime_revision = agent_context.runtime.revision
    existing_runtime_messages = [
        message
        for message in context_messages
        if isinstance(message, Message) and message.metadata.get("runtime_injected")
    ]
    if existing_runtime_messages:
        latest_runtime_message = existing_runtime_messages[-1]
        if latest_runtime_message.metadata.get("runtime_revision") == current_runtime_revision:
            return None

        context_messages[:] = [message for message in context_messages if message not in existing_runtime_messages]
        agent_context.messages = context_messages

    if inspector:
        snapshot_output = await inspector.capture_state(agent_context)
        snapshot_block = normalize_inspector_output(snapshot_output)
    else:
        snapshot_block = build_runtime_snapshot(agent_context, config)

    if snapshot_block is None or not snapshot_block.entries:
        return None

    for entry in snapshot_block.entries:
        if entry.version == 0 and entry.key in agent_context.runtime.variables:
            entry.version = agent_context.runtime.variables[entry.key].version

    runtime_message = Message(
        role="system",
        content_blocks=[snapshot_block],
        metadata={"runtime_injected": True, "runtime_revision": current_runtime_revision},
    )
    context_messages.append(runtime_message)
    agent_context.messages = context_messages
    return runtime_message


async def emit_runtime_message_event(
    runtime_message: Optional[Message],
    parent_id: str,
) -> AsyncGenerator[AgentEvent, None]:
    if runtime_message is None:
        return
    runtime_message_event_id = str(uuid.uuid4())
    yield AgentEvent(
        type="message_start",
        event_id=runtime_message_event_id,
        parent_id=parent_id,
        data={"message": runtime_message},
    )
    yield AgentEvent(
        type="message_end",
        parent_id=runtime_message_event_id,
        data={"message": runtime_message},
    )


def build_worklog_message(
    completed_tool_calls: List[ToolCall],
    committed_ops: List[RuntimeDeltaOp],
    runtime_revision: int,
) -> Message:
    tool_names = ", ".join(tool_call.name for tool_call in completed_tool_calls) or "no tools"
    touched_keys = ", ".join(sorted({op.key for op in committed_ops})) or "no runtime variables"
    return Message(
        role="system",
        content_blocks=[
            TextBlock(
                text=f"worklog: tools=[{tool_names}] runtime_keys=[{touched_keys}] runtime_revision={runtime_revision}",
                metadata={"tag": "worklog", "trajectory": True},
            )
        ],
        metadata={"worklog": True, "trajectory": True},
    )


def commit_runtime_ops(agent_context: AgentContext, ops: List[RuntimeDeltaOp]) -> None:
    for op in ops:
        updated_by = op.metadata.get("source_tool")
        agent_context.runtime.apply_op(op, updated_by=updated_by)
    agent_context.refresh_shared_memory_from_runtime()
