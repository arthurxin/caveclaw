from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..assistant_messages.transform_messages import (
    AssistantFailurePolicy,
    NormalizeToolCallId,
    ProviderStateDropDetail,
    RuntimeBlockConversionDetail,
    SkippedAssistantDetail,
    SyntheticToolResultDetail,
    ToolCallIdRewrite,
    TransformMessagesOptions,
    transform_messages_with_result,
)
from ..assistant_messages.types import AgentMessage, AssistantMessage, Message, ToolCall, ToolResultMessage
from ..llm_provider.compat import requires_thinking_as_text
from ..llm_provider.provider_types import Model
from .session_context import AgentSessionContext

ProviderStatePolicy = Literal["drop", "same-model-only", "preserve"]
RuntimeSnapshotPolicy = Literal["drop", "render-text", "same-model-only", "preserve"]
SyntheticToolResultPolicy = Literal["insert", "skip"]
HandoffRewindPolicy = Literal["none", "previous-round-end"]
HandoffDiagnosticCode = Literal[
    "rewind_applied",
    "provider_state_dropped",
    "runtime_block_converted",
    "synthetic_tool_result_inserted",
    "assistant_message_skipped",
    "tool_call_id_rewritten",
]


@dataclass
class HandoffOptions:
    rewind_policy: Optional[HandoffRewindPolicy] = None
    provider_state_policy: Optional[ProviderStatePolicy] = None
    runtime_snapshot_policy: Optional[RuntimeSnapshotPolicy] = None
    synthetic_tool_result_policy: Optional[SyntheticToolResultPolicy] = None
    assistant_failure_policy: Optional[AssistantFailurePolicy] = None
    normalize_tool_call_id: Optional[NormalizeToolCallId] = None


@dataclass
class ResolvedHandoffOptions:
    rewind_policy: HandoffRewindPolicy
    provider_state_policy: ProviderStatePolicy
    runtime_snapshot_policy: RuntimeSnapshotPolicy
    synthetic_tool_result_policy: SyntheticToolResultPolicy
    assistant_failure_policy: AssistantFailurePolicy
    normalize_tool_call_id: Optional[NormalizeToolCallId] = None


@dataclass
class HandoffDiagnostic:
    code: HandoffDiagnosticCode
    message: str
    message_index: Optional[int] = None
    original_message_count: Optional[int] = None
    rewound_message_count: Optional[int] = None
    rewind_from_index: Optional[int] = None
    role: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    namespaces: List[str] = field(default_factory=list)
    block_type: Optional[str] = None
    mode: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class HandoffResult:
    target_model: Model
    messages: List[AgentMessage]
    resolved_options: ResolvedHandoffOptions
    session_context: Optional[AgentSessionContext] = None
    rewind_applied: bool = False
    original_message_count: int = 0
    rewound_message_count: int = 0
    rewind_from_index: Optional[int] = None
    tool_call_id_map: dict[str, str] = field(default_factory=dict)
    synthetic_tool_results_added: int = 0
    skipped_assistant_messages: int = 0
    provider_state_drops: int = 0
    runtime_snapshot_conversions: int = 0
    tool_call_id_rewrites: List[ToolCallIdRewrite] = field(default_factory=list)
    synthetic_tool_result_details: List[SyntheticToolResultDetail] = field(default_factory=list)
    skipped_assistant_details: List[SkippedAssistantDetail] = field(default_factory=list)
    provider_state_drop_details: List[ProviderStateDropDetail] = field(default_factory=list)
    runtime_block_conversion_details: List[RuntimeBlockConversionDetail] = field(default_factory=list)
    diagnostics: List[HandoffDiagnostic] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class _RewindSelection:
    messages: List[AgentMessage]
    rewind_applied: bool
    original_message_count: int
    rewound_message_count: int
    rewind_from_index: Optional[int]


def handoff_messages(
    messages: List[AgentMessage],
    target_model: Model,
    *,
    options: Optional[HandoffOptions] = None,
) -> HandoffResult:
    resolved_options = resolve_handoff_options(target_model, options=options)
    rewind_selection = _select_messages_for_handoff(
        messages,
        rewind_policy=resolved_options.rewind_policy,
    )
    cloned_messages: List[AgentMessage] = []
    llm_messages: List[Message] = []
    llm_positions: List[int] = []

    for index, message in enumerate(rewind_selection.messages):
        if isinstance(message, Message):
            cloned = clone_message_for_handoff(
                message,
                preserve_provider_state=True,
            )
            cloned_messages.append(cloned)
            llm_messages.append(cloned)
            llm_positions.append(index)
        else:
            cloned_messages.append(message)

    transform_result = transform_messages_with_result(
        llm_messages,
        target_model,
        options=TransformMessagesOptions(
            provider_state_policy=resolved_options.provider_state_policy,
            runtime_snapshot_policy=resolved_options.runtime_snapshot_policy,
            synthetic_tool_result_policy=resolved_options.synthetic_tool_result_policy,
            assistant_failure_policy=resolved_options.assistant_failure_policy,
            normalize_tool_call_id=resolved_options.normalize_tool_call_id,
        ),
    )
    for index, message in zip(llm_positions, transform_result.messages):
        cloned_messages[index] = message
    diagnostics = build_handoff_diagnostics(
        transform_result,
        rewind_applied=rewind_selection.rewind_applied,
        original_message_count=rewind_selection.original_message_count,
        rewound_message_count=rewind_selection.rewound_message_count,
        rewind_from_index=rewind_selection.rewind_from_index,
    )
    warnings = [diagnostic.message for diagnostic in diagnostics]
    return HandoffResult(
        target_model=target_model,
        messages=cloned_messages,
        resolved_options=resolved_options,
        rewind_applied=rewind_selection.rewind_applied,
        original_message_count=rewind_selection.original_message_count,
        rewound_message_count=rewind_selection.rewound_message_count,
        rewind_from_index=rewind_selection.rewind_from_index,
        tool_call_id_map=dict(transform_result.tool_call_id_map),
        synthetic_tool_results_added=transform_result.synthetic_tool_results_added,
        skipped_assistant_messages=transform_result.skipped_assistant_messages,
        provider_state_drops=transform_result.provider_state_drops,
        runtime_snapshot_conversions=transform_result.runtime_snapshot_conversions,
        tool_call_id_rewrites=list(transform_result.tool_call_id_rewrites),
        synthetic_tool_result_details=list(transform_result.synthetic_tool_result_details),
        skipped_assistant_details=list(transform_result.skipped_assistant_details),
        provider_state_drop_details=list(transform_result.provider_state_drop_details),
        runtime_block_conversion_details=list(transform_result.runtime_block_conversion_details),
        diagnostics=diagnostics,
        warnings=warnings,
    )


def handoff_session_context(
    session_context: AgentSessionContext,
    target_model: Model,
    *,
    options: Optional[HandoffOptions] = None,
) -> HandoffResult:
    message_result = handoff_messages(session_context.messages, target_model, options=options)
    handed_off_context = AgentSessionContext(
        messages=list(message_result.messages),
        tools=list(session_context.tools),
        host=session_context.host,
        metadata=dict(session_context.metadata),
    )
    return HandoffResult(
        target_model=target_model,
        messages=list(message_result.messages),
        resolved_options=message_result.resolved_options,
        session_context=handed_off_context,
        rewind_applied=message_result.rewind_applied,
        original_message_count=message_result.original_message_count,
        rewound_message_count=message_result.rewound_message_count,
        rewind_from_index=message_result.rewind_from_index,
        tool_call_id_map=dict(message_result.tool_call_id_map),
        synthetic_tool_results_added=message_result.synthetic_tool_results_added,
        skipped_assistant_messages=message_result.skipped_assistant_messages,
        provider_state_drops=message_result.provider_state_drops,
        runtime_snapshot_conversions=message_result.runtime_snapshot_conversions,
        tool_call_id_rewrites=list(message_result.tool_call_id_rewrites),
        synthetic_tool_result_details=list(message_result.synthetic_tool_result_details),
        skipped_assistant_details=list(message_result.skipped_assistant_details),
        provider_state_drop_details=list(message_result.provider_state_drop_details),
        runtime_block_conversion_details=list(message_result.runtime_block_conversion_details),
        diagnostics=list(message_result.diagnostics),
        warnings=list(message_result.warnings),
    )


def resolve_handoff_options(
    target_model: Model,
    *,
    options: Optional[HandoffOptions] = None,
) -> ResolvedHandoffOptions:
    requested = options or HandoffOptions()
    thinking_as_text = requires_thinking_as_text(target_model)

    rewind_policy = requested.rewind_policy or "previous-round-end"
    provider_state_policy = requested.provider_state_policy
    if provider_state_policy is None:
        provider_state_policy = "drop" if thinking_as_text else "same-model-only"

    runtime_snapshot_policy = requested.runtime_snapshot_policy
    if runtime_snapshot_policy is None:
        runtime_snapshot_policy = "render-text" if thinking_as_text else "same-model-only"

    synthetic_tool_result_policy = requested.synthetic_tool_result_policy or "insert"
    assistant_failure_policy = requested.assistant_failure_policy or "skip"

    return ResolvedHandoffOptions(
        rewind_policy=rewind_policy,
        provider_state_policy=provider_state_policy,
        runtime_snapshot_policy=runtime_snapshot_policy,
        synthetic_tool_result_policy=synthetic_tool_result_policy,
        assistant_failure_policy=assistant_failure_policy,
        normalize_tool_call_id=requested.normalize_tool_call_id,
    )


def clone_message_for_handoff(message: Message, *, preserve_provider_state: bool) -> Message:
    base_kwargs = {
        "role": message.role,
        "content_blocks": list(message.content_blocks),
        "raw_content": message.raw_content,
        "provider_state": dict(message.provider_state) if (preserve_provider_state and message.provider_state) else None,
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


def _select_messages_for_handoff(
    messages: List[AgentMessage],
    *,
    rewind_policy: HandoffRewindPolicy,
) -> _RewindSelection:
    original_message_count = len(messages)
    if rewind_policy == "none" or not messages:
        return _RewindSelection(
            messages=list(messages),
            rewind_applied=False,
            original_message_count=original_message_count,
            rewound_message_count=original_message_count,
            rewind_from_index=None,
        )

    last_safe_index: Optional[int] = None
    safe_prefix = True
    saw_llm_message = False

    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            continue
        saw_llm_message = True

        if isinstance(message, ToolResultMessage):
            safe_prefix = False
            continue

        if isinstance(message, AssistantMessage):
            if _is_safe_handoff_endpoint(message):
                safe_prefix = True
                last_safe_index = index
            else:
                safe_prefix = False
            continue

        if safe_prefix:
            last_safe_index = index

    if safe_prefix or not saw_llm_message:
        return _RewindSelection(
            messages=list(messages),
            rewind_applied=False,
            original_message_count=original_message_count,
            rewound_message_count=original_message_count,
            rewind_from_index=None,
        )

    rewound_message_count = 0 if last_safe_index is None else last_safe_index + 1
    return _RewindSelection(
        messages=list(messages[:rewound_message_count]),
        rewind_applied=True,
        original_message_count=original_message_count,
        rewound_message_count=rewound_message_count,
        rewind_from_index=rewound_message_count,
    )


def _is_safe_handoff_endpoint(message: AssistantMessage) -> bool:
    if message.tool_calls:
        return False
    return message.stop_reason not in {"tool_use", "error", "aborted"}


def build_handoff_diagnostics(
    transform_result,
    *,
    rewind_applied: bool = False,
    original_message_count: Optional[int] = None,
    rewound_message_count: Optional[int] = None,
    rewind_from_index: Optional[int] = None,
) -> List[HandoffDiagnostic]:
    diagnostics: List[HandoffDiagnostic] = []

    if rewind_applied:
        diagnostics.append(
            HandoffDiagnostic(
                code="rewind_applied",
                message=(
                    f"rewound transcript to safe point from {original_message_count} to "
                    f"{rewound_message_count} messages"
                ),
                message_index=rewind_from_index,
                original_message_count=original_message_count,
                rewound_message_count=rewound_message_count,
                rewind_from_index=rewind_from_index,
            )
        )

    for detail in transform_result.provider_state_drop_details:
        diagnostics.append(
            HandoffDiagnostic(
                code="provider_state_dropped",
                message=f"dropped provider_state namespaces={','.join(detail.namespaces)}",
                message_index=detail.message_index,
                role=detail.role,
                namespaces=list(detail.namespaces),
            )
        )

    for detail in transform_result.runtime_block_conversion_details:
        diagnostics.append(
            HandoffDiagnostic(
                code="runtime_block_converted",
                message=f"converted {detail.block_type} via {detail.mode}",
                message_index=detail.message_index,
                role=detail.role,
                block_type=detail.block_type,
                mode=detail.mode,
            )
        )

    for detail in transform_result.synthetic_tool_result_details:
        diagnostics.append(
            HandoffDiagnostic(
                code="synthetic_tool_result_inserted",
                message=f"inserted synthetic tool result for {detail.tool_name}",
                message_index=detail.message_index,
                tool_name=detail.tool_name,
                tool_call_id=detail.tool_call_id,
            )
        )

    for detail in transform_result.skipped_assistant_details:
        diagnostics.append(
            HandoffDiagnostic(
                code="assistant_message_skipped",
                message=f"skipped assistant message with stop_reason={detail.stop_reason}",
                message_index=detail.message_index,
                stop_reason=detail.stop_reason,
            )
        )

    for detail in transform_result.tool_call_id_rewrites:
        diagnostics.append(
            HandoffDiagnostic(
                code="tool_call_id_rewritten",
                message=f"rewrote tool call id for {detail.tool_name}",
                message_index=detail.message_index,
                tool_name=detail.tool_name,
                source_id=detail.source_id,
                target_id=detail.target_id,
            )
        )

    return diagnostics


__all__ = [
    "HandoffDiagnostic",
    "HandoffDiagnosticCode",
    "HandoffOptions",
    "HandoffRewindPolicy",
    "HandoffResult",
    "ProviderStatePolicy",
    "ResolvedHandoffOptions",
    "RuntimeSnapshotPolicy",
    "SyntheticToolResultPolicy",
    "build_handoff_diagnostics",
    "clone_message_for_handoff",
    "handoff_messages",
    "handoff_session_context",
    "resolve_handoff_options",
]
