from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ...assistant_messages.types import AgentContext, AgentEvent, AgentLoopConfig, AssistantMessage, Message, TextBlock
from .bridge import PythonRuntimeBridgeResult, resolve_python_runtime_bridge
from .executor import IPythonProgramRuntime, PythonProgramExecutor
from .parser import extract_first_python_program_block
from .types import PythonProgramBlock, PythonProgramExecutionRequest, PythonProgramExecutionResult


@dataclass
class PythonProgramLaneExecution:
    result: PythonProgramExecutionResult
    events: List[AgentEvent] = field(default_factory=list)
    result_message: Optional[Message] = None
    worklog_message: Optional[Message] = None
    runtime_bridge_result: Optional[PythonRuntimeBridgeResult] = None


class PythonProgramExecutionController(Protocol):
    enabled: bool

    def select_python_block(self, assistant_message: AssistantMessage) -> Optional[PythonProgramBlock]:
        ...


@dataclass
class DefaultPythonProgramExecutionController:
    enabled: bool = False
    require_no_tool_calls: bool = True

    def select_python_block(self, assistant_message: AssistantMessage) -> Optional[PythonProgramBlock]:
        if not self.enabled:
            return None
        if self.require_no_tool_calls and assistant_message.tool_calls:
            return None
        return extract_first_python_program_block(assistant_message.content)


def sanitize_assistant_message_for_python_execution(
    assistant_message: AssistantMessage,
    python_block: PythonProgramBlock,
) -> AssistantMessage:
    if python_block.raw_fence:
        sanitized_content = python_block.raw_fence
    else:
        sanitized_content = f"```{python_block.language}\n{python_block.code}\n```"
    return AssistantMessage(
        content=sanitized_content,
        raw_content=sanitized_content,
        tool_calls=list(assistant_message.tool_calls) if assistant_message.tool_calls else None,
        stop_reason=assistant_message.stop_reason,
        model=assistant_message.model,
        provider=assistant_message.provider,
        api=assistant_message.api,
        usage=dict(assistant_message.usage) if assistant_message.usage else None,
        provider_state=dict(assistant_message.provider_state) if assistant_message.provider_state else None,
        metadata=dict(assistant_message.metadata),
        timestamp=assistant_message.timestamp,
    )


def _resolve_backend(config: AgentLoopConfig) -> str:
    backend = getattr(config, "python_program_backend", None)
    if backend in {"python", "ipython"}:
        return backend
    return "ipython"


def _resolve_executor(config: AgentLoopConfig) -> PythonProgramExecutor:
    configured_executor = getattr(config, "python_program_executor", None)
    if configured_executor is not None:
        return configured_executor
    return PythonProgramExecutor()


async def _build_namespace(
    agent_context: AgentContext,
    assistant_message: AssistantMessage,
    python_block: PythonProgramBlock,
    config: AgentLoopConfig,
) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "runtime": agent_context.runtime,
        "python_runtime": agent_context.python_runtime,
        "shared_memory": agent_context.shared_memory,
        "messages": list(agent_context.messages),
        "assistant_message": assistant_message,
        "python_block": python_block,
    }

    namespace_builder = getattr(config, "build_python_program_namespace", None)
    if namespace_builder is None:
        return namespace

    extra_namespace = await namespace_builder(agent_context, assistant_message, python_block)
    if extra_namespace:
        namespace.update(extra_namespace)
    return namespace


def build_python_program_result_message(
    python_block: PythonProgramBlock,
    result: PythonProgramExecutionResult,
    *,
    synced_variables: Optional[List[str]] = None,
) -> Message:
    success_instruction = (
        "Local execution has already finished. "
        "Based on the authoritative stdout and runtime snapshot below, answer the original user request directly now in plain language. "
        "Do not say that you cannot access local files after receiving this local execution result."
    )
    failure_instruction = (
        "Local execution failed. "
        "Use stdout, stderr, error, and the runtime snapshot below to correct the next python block. "
        "Do not ignore local execution feedback."
    )
    parts = [
        "LOCAL PYTHON EXECUTION RESULT: The fenced python block was executed inside the user's local runtime.",
        "Treat stdout, stderr, runtime snapshots, and synced variables as authoritative local facts.",
        success_instruction if result.success else failure_instruction,
        f"python program execution: backend={result.backend} success={result.success}",
    ]
    synced = ", ".join(synced_variables or []) or "none"
    parts.append(f"synced_variables: {synced}")
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        parts.append("stdout:\n" + stdout)
    if stderr:
        parts.append("stderr:\n" + stderr)
    if result.error:
        parts.append("error:\n" + result.error)

    return Message(
        role="user",
        content_blocks=[TextBlock(text="\n\n".join(parts), metadata={"tag": "python_program_execution"})],
        metadata={
            "python_program_execution": True,
            "python_program_backend": result.backend,
            "python_program_success": result.success,
            "python_program_language": python_block.language,
        },
    )


def build_python_program_worklog_message(
    result: PythonProgramExecutionResult,
    runtime_revision: int,
    *,
    synced_variables: Optional[List[str]] = None,
) -> Message:
    synced = ", ".join(synced_variables or []) or "none"
    return Message(
        role="system",
        content_blocks=[
            TextBlock(
                text=(
                    "worklog: python_program "
                    f"backend=[{result.backend}] success=[{str(result.success).lower()}] "
                    f"runtime_revision={runtime_revision} synced_variables=[{synced}]"
                ),
                metadata={"tag": "worklog", "trajectory": True, "lane": "python_program"},
            )
        ],
        metadata={"worklog": True, "trajectory": True, "lane": "python_program"},
    )


def build_python_program_followup_message(result: PythonProgramExecutionResult) -> Message:
    del result
    raise NotImplementedError("python followup messages are no longer used")


def resolve_python_program_execution_controller(config: AgentLoopConfig) -> PythonProgramExecutionController:
    configured = getattr(config, "python_program_execution", None)
    if configured is None:
        return DefaultPythonProgramExecutionController(enabled=False)
    if configured is False:
        return DefaultPythonProgramExecutionController(enabled=False)
    if configured is True:
        return DefaultPythonProgramExecutionController(enabled=True)
    if configured is not None and hasattr(configured, "select_python_block"):
        return configured
    return DefaultPythonProgramExecutionController(enabled=True)


def is_python_program_execution_enabled(config: AgentLoopConfig) -> bool:
    configured = getattr(config, "python_program_execution", None)
    if configured is None or configured is False:
        return False
    if configured is True:
        return True
    return bool(getattr(configured, "enabled", True))


async def execute_python_program_lane(
    assistant_message: AssistantMessage,
    python_block: PythonProgramBlock,
    agent_context: AgentContext,
    config: AgentLoopConfig,
    turn_event_id: str,
) -> PythonProgramLaneExecution:
    backend = _resolve_backend(config)
    namespace = await _build_namespace(agent_context, assistant_message, python_block, config)
    executor = (
        _resolve_executor(config)
        if agent_context.python_runtime is None
        else PythonProgramExecutor(ipython_runtime=agent_context.python_runtime)
    )
    lane_event_id = str(uuid.uuid4())
    events = [
        AgentEvent(
            type="python_program_execution_start",
            event_id=lane_event_id,
            parent_id=turn_event_id,
            data={
                "backend": backend,
                "language": python_block.language,
                "code": python_block.code,
                "namespace_keys": sorted(namespace.keys()),
            },
        )
    ]

    execution_request = PythonProgramExecutionRequest(
        block=python_block,
        backend=backend,
        namespace=namespace,
        metadata={"assistant_message_metadata": dict(assistant_message.metadata)},
    )
    result = await asyncio.to_thread(
        executor.execute,
        execution_request,
    )
    runtime_bridge = resolve_python_runtime_bridge(config)
    runtime_bridge_result = await runtime_bridge.build_runtime_bridge_result(
        execution_request=execution_request,
        execution_result=result,
        agent_context=agent_context,
        python_block=python_block,
    )
    if runtime_bridge_result.runtime_ops:
        for op in runtime_bridge_result.runtime_ops:
            agent_context.runtime.apply_op(op, updated_by="python_program")
        agent_context.refresh_shared_memory_from_runtime()
    result_message = build_python_program_result_message(
        python_block,
        result,
        synced_variables=runtime_bridge_result.synced_variables,
    )
    worklog_message = None

    result_message_event_id = str(uuid.uuid4())
    events.append(
        AgentEvent(
            type="message_start",
            event_id=result_message_event_id,
            parent_id=lane_event_id,
            data={"message": result_message},
        )
    )
    events.append(
        AgentEvent(
            type="message_end",
            parent_id=result_message_event_id,
            data={"message": result_message},
        )
    )
    events.append(
        AgentEvent(
            type="python_program_execution_success" if result.success else "python_program_execution_error",
            parent_id=lane_event_id,
            data={
                "result": result,
                "runtime_revision": agent_context.runtime.revision,
                "synced_variables": list(runtime_bridge_result.synced_variables),
                "skipped_variables": dict(runtime_bridge_result.skipped_variables),
                "namespace_before_keys": list(runtime_bridge_result.namespace_before_keys),
                "namespace_after_keys": list(runtime_bridge_result.namespace_after_keys),
                "runtime_bridge_result": runtime_bridge_result,
            },
        )
    )

    return PythonProgramLaneExecution(
        result=result,
        events=events,
        result_message=result_message,
        worklog_message=worklog_message,
        runtime_bridge_result=runtime_bridge_result,
    )


__all__ = [
    "DefaultPythonProgramExecutionController",
    "PythonProgramLaneExecution",
    "PythonProgramExecutionController",
    "build_python_program_result_message",
    "build_python_program_worklog_message",
    "execute_python_program_lane",
    "is_python_program_execution_enabled",
    "resolve_python_program_execution_controller",
]
