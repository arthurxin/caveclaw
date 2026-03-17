from __future__ import annotations

import os
import uuid
from typing import Any, AsyncGenerator, Callable, List, Optional

from dotenv import load_dotenv

from .assistant_stream import stream_assistant_response
from .compaction import compact_messages_for_llm
from .runtime_projection import (
    build_worklog_message,
    commit_runtime_ops,
    emit_runtime_message_event,
    inject_runtime_snapshot,
)
from .tool_execution import collect_tool_result_runtime_ops, execute_tool_with_streaming_updates
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    BasicCancellationSignal,
    CancellationSignal,
    Message,
    RuntimeDeltaOp,
    ToolResultMessage,
    RuntimeState,
    content_blocks_from_text,
)

# Load environment variables
load_dotenv()


async def run_loop(
    context_messages: List[AgentMessage],
    tools: List[AgentTool],
    config: AgentLoopConfig,
    stream_fn: Optional[Callable[..., AsyncGenerator[Any, None]]] = None,
    runtime: Optional[RuntimeState] = None,
) -> AsyncGenerator[AgentEvent, None]:
    """
    The core state machine loop with streaming assistant lifecycle events,
    runtime injection, and tool runtime commits.
    """

    env_max_rounds = int(os.environ.get("MAX_ROUNDS", "20"))
    env_max_consec_failures = int(os.environ.get("MAX_CONSECUTIVE_FAILURES", "5"))

    max_rounds = getattr(config, "max_rounds", env_max_rounds) or env_max_rounds
    max_consecutive_failures = (
        getattr(config, "max_consecutive_tool_failures", env_max_consec_failures) or env_max_consec_failures
    )

    agent_context = AgentContext(messages=context_messages, runtime=runtime or RuntimeState(), shared_memory={})
    abort_signal: CancellationSignal = getattr(config, "abort_signal", None) or BasicCancellationSignal()
    loop_event_id = str(uuid.uuid4())
    yield AgentEvent(type="agent_start", event_id=loop_event_id)

    round_count = 0
    consecutive_failures = 0

    while True:
        if abort_signal.is_cancelled:
            break

        runtime_message = await inject_runtime_snapshot(context_messages, agent_context, config)
        async for event in emit_runtime_message_event(runtime_message, loop_event_id):
            yield event

        round_count += 1
        turn_event_id = str(uuid.uuid4())

        if round_count > max_rounds:
            consolidation_handler = getattr(config, "handle_consolidation", None)
            if consolidation_handler:
                consolidated_messages = await consolidation_handler(context_messages, agent_context)
                if consolidated_messages:
                    context_messages = list(consolidated_messages)
                    agent_context.messages = context_messages
                    runtime_message = await inject_runtime_snapshot(context_messages, agent_context, config)
                    async for event in emit_runtime_message_event(runtime_message, loop_event_id):
                        yield event
                    yield AgentEvent(
                        type="consolidation_applied",
                        parent_id=loop_event_id,
                        data={"message_count": len(context_messages), "runtime_revision": agent_context.runtime.revision},
                    )
                    round_count = 0
                    continue
            yield AgentEvent(
                type="consolidation_required",
                parent_id=loop_event_id,
                data={"reason": "max_rounds_reached", "max_rounds": max_rounds},
            )
            yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
            return

        if consecutive_failures >= max_consecutive_failures:
            yield AgentEvent(
                type="human_intervention_required",
                parent_id=loop_event_id,
                data={"reason": "max_consecutive_failures", "consecutive_failures": consecutive_failures},
            )
            yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
            return

        if hasattr(config, "transform_context"):
            context_messages = await config.transform_context(context_messages)
            agent_context.messages = context_messages

        compacted_messages = await compact_messages_for_llm(context_messages, agent_context, config)
        llm_messages = await config.convert_to_llm(compacted_messages)

        yield AgentEvent(
            type="turn_start",
            event_id=turn_event_id,
            parent_id=loop_event_id,
            data={"round": round_count, "runtime_revision": agent_context.runtime.revision},
        )

        pending_events: List[AgentEvent] = []
        assistant_msg = await stream_assistant_response(
            llm_messages,
            tools,
            config,
            stream_fn=stream_fn,
            yield_event=pending_events.append,
            message_parent_id=turn_event_id,
            signal=abort_signal,
        )
        for event in pending_events:
            yield event

        context_messages.append(assistant_msg)
        agent_context.messages = context_messages

        if assistant_msg.stop_reason == "aborted" or abort_signal.is_cancelled:
            yield AgentEvent(
                type="turn_end",
                parent_id=turn_event_id,
                data={"message": assistant_msg, "tool_results": [], "aborted": True},
            )
            yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
            return

        if not assistant_msg.tool_calls:
            yield AgentEvent(
                type="turn_end",
                parent_id=turn_event_id,
                data={"message": assistant_msg, "tool_results": []},
            )
            consecutive_failures = 0

            if hasattr(config, "get_followup_messages"):
                followup = await config.get_followup_messages()
                if followup:
                    context_messages.extend(followup)
                    agent_context.messages = context_messages
                    continue
            break

        steering_interrupted = False
        tool_results: List[ToolResultMessage] = []
        batch_runtime_ops: List[RuntimeDeltaOp] = []

        for tool_call in assistant_msg.tool_calls:
            tool = next((candidate for candidate in tools if candidate.name == tool_call.name), None)
            runtime_selection = tool.resolve_runtime_selection(agent_context, tool_call.arguments) if tool else None
            tool_event_id = str(uuid.uuid4())
            yield AgentEvent(
                type="tool_execution_start",
                event_id=tool_event_id,
                parent_id=turn_event_id,
                data={
                    "tool_call": tool_call,
                    "reads": getattr(tool, "reads", []),
                    "writes": getattr(tool, "writes", []),
                    "temp_outputs": getattr(tool, "temp_outputs", []),
                    "runtime_selection": runtime_selection.to_payload() if runtime_selection else None,
                },
            )

            if not tool:
                error_msg = f"Error: Tool {tool_call.name} not found."
                tool_result_message = ToolResultMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=error_msg,
                    is_error=True,
                )
                context_messages.append(tool_result_message)
                agent_context.messages = context_messages
                tool_results.append(tool_result_message)
                consecutive_failures += 1
                yield AgentEvent(type="message_start", parent_id=tool_event_id, data={"message": tool_result_message})
                yield AgentEvent(type="message_end", parent_id=tool_event_id, data={"message": tool_result_message})
                yield AgentEvent(type="tool_execution_error", parent_id=tool_event_id, data={"error": error_msg})
            else:
                try:
                    execution = await execute_tool_with_streaming_updates(
                        tool,
                        tool_call,
                        agent_context,
                        abort_signal,
                        tool_event_id,
                        runtime_selection=runtime_selection,
                    )
                    for event in execution.events:
                        yield event

                    result = execution.result
                    batch_runtime_ops.extend(execution.staged_runtime_ops)
                    batch_runtime_ops.extend(collect_tool_result_runtime_ops(tool.name, result))

                    result_blocks = (
                        list(result.content_blocks) if result.content_blocks else content_blocks_from_text(result.content)
                    )
                    tool_result_message = ToolResultMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content_blocks=result_blocks,
                        raw_content=result.raw_content if result.raw_content is not None else None,
                        details=result.details,
                    )
                    context_messages.append(tool_result_message)
                    agent_context.messages = context_messages
                    tool_results.append(tool_result_message)
                    consecutive_failures = 0

                    yield AgentEvent(type="message_start", parent_id=tool_event_id, data={"message": tool_result_message})
                    yield AgentEvent(type="message_end", parent_id=tool_event_id, data={"message": tool_result_message})
                    yield AgentEvent(
                        type="tool_execution_success",
                        parent_id=tool_event_id,
                        data={
                            "result": tool_result_message.content,
                            "delta_staged": bool(execution.staged_runtime_ops or result.state_delta or result.runtime_ops),
                            "runtime_revision": agent_context.runtime.revision,
                        },
                    )
                    if abort_signal.is_cancelled:
                        break
                except Exception as error:
                    error_msg = f"Error executing tool {tool_call.name}: {str(error)}"
                    tool_result_message = ToolResultMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=error_msg,
                        is_error=True,
                    )
                    context_messages.append(tool_result_message)
                    agent_context.messages = context_messages
                    tool_results.append(tool_result_message)
                    consecutive_failures += 1
                    yield AgentEvent(type="message_start", parent_id=tool_event_id, data={"message": tool_result_message})
                    yield AgentEvent(type="message_end", parent_id=tool_event_id, data={"message": tool_result_message})
                    yield AgentEvent(type="tool_execution_error", parent_id=tool_event_id, data={"error": error_msg})
                    if abort_signal.is_cancelled:
                        break

            if hasattr(config, "get_steering_messages"):
                steering = await config.get_steering_messages()
                if steering:
                    context_messages.extend(steering)
                    agent_context.messages = context_messages
                    steering_interrupted = True
                    yield AgentEvent(
                        type="steering_interruption",
                        parent_id=turn_event_id,
                        data={"steering_messages": steering},
                    )
                    break

            if abort_signal.is_cancelled:
                break

        if batch_runtime_ops:
            commit_runtime_ops(agent_context, batch_runtime_ops)

        runtime_message = await inject_runtime_snapshot(context_messages, agent_context, config)
        async for event in emit_runtime_message_event(runtime_message, turn_event_id):
            yield event

        worklog_message = build_worklog_message(
            list(assistant_msg.tool_calls),
            batch_runtime_ops,
            agent_context.runtime.revision,
        )
        context_messages.append(worklog_message)
        agent_context.messages = context_messages
        worklog_event_id = str(uuid.uuid4())
        yield AgentEvent(
            type="message_start",
            event_id=worklog_event_id,
            parent_id=turn_event_id,
            data={"message": worklog_message},
        )
        yield AgentEvent(
            type="message_end",
            parent_id=worklog_event_id,
            data={"message": worklog_message},
        )

        yield AgentEvent(
            type="turn_end",
            parent_id=turn_event_id,
            data={
                "message": assistant_msg,
                "tools_executed": len(assistant_msg.tool_calls),
                "interrupted": steering_interrupted,
                "tool_results": tool_results,
                "batch_runtime_ops": batch_runtime_ops,
                "runtime_revision": agent_context.runtime.revision,
            },
        )

        if abort_signal.is_cancelled:
            yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
            return
        if steering_interrupted:
            continue

    yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
