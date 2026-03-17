from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..assistant_messages.types import (
    AgentContext,
    AgentEvent,
    AgentTool,
    AgentToolUpdate,
    CancellationSignal,
    RuntimeDeltaOp,
    ToolCall,
    ToolResult,
    ToolRuntimeSelection,
)


@dataclass
class StreamedToolExecution:
    result: ToolResult
    events: List[AgentEvent] = field(default_factory=list)
    staged_runtime_ops: List[RuntimeDeltaOp] = field(default_factory=list)
    runtime_selection: Optional[ToolRuntimeSelection] = None


def normalize_update(update: Any) -> AgentToolUpdate:
    if isinstance(update, AgentToolUpdate):
        return update
    if isinstance(update, ToolResult):
        return AgentToolUpdate(
            content_blocks=list(update.content_blocks),
            raw_content=update.raw_content,
            details=update.details,
            runtime_ops=list(update.runtime_ops),
            metadata=dict(update.metadata),
        )
    if isinstance(update, str):
        return AgentToolUpdate(content=update)
    if hasattr(update, "content") and hasattr(update, "details"):
        return AgentToolUpdate(content=getattr(update, "content", ""), details=getattr(update, "details", None))
    return AgentToolUpdate(content=str(update))


def collect_tool_result_runtime_ops(tool_name: str, result: ToolResult) -> List[RuntimeDeltaOp]:
    ops: List[RuntimeDeltaOp] = []
    if result.state_delta:
        for key, value in result.state_delta.items():
            ops.append(RuntimeDeltaOp(op="set", key=key, value=value, metadata={"source_tool": tool_name}))
    ops.extend(result.runtime_ops)
    return ops


async def invoke_tool_execute(
    tool: AgentTool,
    tool_call: ToolCall,
    agent_context: AgentContext,
    signal: CancellationSignal,
    on_update: Callable[[AgentToolUpdate], None],
    runtime_selection: Optional[ToolRuntimeSelection] = None,
) -> ToolResult:
    execute = tool.execute
    signature = inspect.signature(execute)
    kwargs: Dict[str, Any] = {}
    runtime_selection = runtime_selection or tool.resolve_runtime_selection(agent_context, tool_call.arguments)
    previous_selection = agent_context.active_tool_selection
    agent_context.active_tool_selection = runtime_selection
    if "on_update" in signature.parameters:
        kwargs["on_update"] = on_update
    if "signal" in signature.parameters:
        kwargs["signal"] = signal
    try:
        return await execute(tool_call.id, tool_call.arguments, agent_context, **kwargs)
    finally:
        agent_context.active_tool_selection = previous_selection


async def execute_tool_with_streaming_updates(
    tool: AgentTool,
    tool_call: ToolCall,
    agent_context: AgentContext,
    signal: CancellationSignal,
    tool_event_id: str,
    runtime_selection: Optional[ToolRuntimeSelection] = None,
) -> StreamedToolExecution:
    updates_queue: asyncio.Queue[Optional[AgentToolUpdate]] = asyncio.Queue()
    streamed_events: List[AgentEvent] = []
    staged_runtime_ops: List[RuntimeDeltaOp] = []
    tool_update_event_id = str(uuid.uuid4())

    def on_update(partial_update: AgentToolUpdate) -> None:
        updates_queue.put_nowait(normalize_update(partial_update))

    async def drain_updates_until_done(task: "asyncio.Task[ToolResult]") -> None:
        while True:
            if task.done() and updates_queue.empty():
                break
            try:
                update = await asyncio.wait_for(updates_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if update is None:
                continue
            if update.runtime_ops:
                staged_runtime_ops.extend(update.runtime_ops)
            streamed_events.append(
                AgentEvent(
                    type="tool_execution_update",
                    event_id=tool_update_event_id,
                    parent_id=tool_event_id,
                    data={"tool_call": tool_call, "partial_result": update},
                )
            )

    runtime_selection = runtime_selection or tool.resolve_runtime_selection(agent_context, tool_call.arguments)

    task = asyncio.create_task(
        invoke_tool_execute(
            tool,
            tool_call,
            agent_context,
            signal,
            on_update,
            runtime_selection=runtime_selection,
        )
    )
    await drain_updates_until_done(task)
    result = await task
    return StreamedToolExecution(
        result=result,
        events=streamed_events,
        staged_runtime_ops=staged_runtime_ops,
        runtime_selection=runtime_selection,
    )
