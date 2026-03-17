from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Union


ContentBlockType = Literal[
    "text",
    "image",
    "thinking",
    "tool_call",
    "tool_result",
    "runtime_ref",
    "runtime_snapshot",
]
RuntimeValueKind = Literal["opaque", "structured", "tabular", "message_blocks", "binary_ref"]
RuntimeOperationType = Literal["set", "delete", "merge", "append", "touch", "replace_blocks"]


@dataclass
class BaseContentBlock:
    type: ContentBlockType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"type": self.type}
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class TextBlock(BaseContentBlock):
    text: str = ""
    type: Literal["text"] = "text"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["text"] = self.text
        return payload


@dataclass
class ImageBlock(BaseContentBlock):
    image_url: Optional[str] = None
    mime_type: Optional[str] = None
    alt_text: Optional[str] = None
    type: Literal["image"] = "image"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        if self.image_url is not None:
            payload["image_url"] = self.image_url
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        if self.alt_text is not None:
            payload["alt_text"] = self.alt_text
        return payload


@dataclass
class ThinkingBlock(BaseContentBlock):
    thinking: str = ""
    signature: Optional[str] = None
    type: Literal["thinking"] = "thinking"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["thinking"] = self.thinking
        if self.signature is not None:
            payload["signature"] = self.signature
        return payload


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class ToolCallBlock(BaseContentBlock):
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["id"] = self.id
        payload["name"] = self.name
        payload["arguments"] = self.arguments
        return payload


@dataclass
class ToolResultBlock(BaseContentBlock):
    tool_call_id: str = ""
    tool_name: str = ""
    content_blocks: List["ContentBlock"] = field(default_factory=list)
    is_error: bool = False
    details: Any = None
    type: Literal["tool_result"] = "tool_result"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["tool_call_id"] = self.tool_call_id
        payload["tool_name"] = self.tool_name
        payload["content"] = [block.to_dict() for block in self.content_blocks]
        payload["is_error"] = self.is_error
        if self.details is not None:
            payload["details"] = self.details
        return payload


@dataclass
class RuntimeRefBlock(BaseContentBlock):
    key: str = ""
    version: Optional[int] = None
    label: Optional[str] = None
    type: Literal["runtime_ref"] = "runtime_ref"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["key"] = self.key
        if self.version is not None:
            payload["version"] = self.version
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass
class RuntimeSnapshotEntry:
    key: str
    version: int
    summary_blocks: List["ContentBlock"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "key": self.key,
            "version": self.version,
            "summary": [block.to_dict() for block in self.summary_blocks],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class RuntimeSnapshotBlock(BaseContentBlock):
    entries: List[RuntimeSnapshotEntry] = field(default_factory=list)
    type: Literal["runtime_snapshot"] = "runtime_snapshot"

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


ContentBlock = Union[
    TextBlock,
    ImageBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    RuntimeRefBlock,
    RuntimeSnapshotBlock,
]


def text_block(text: str, **metadata: Any) -> TextBlock:
    return TextBlock(text=text, metadata=metadata)


def content_blocks_from_text(text: str) -> List[ContentBlock]:
    if not text:
        return []
    return [TextBlock(text=text)]


def content_blocks_to_text(content_blocks: List[ContentBlock]) -> str:
    text_parts: List[str] = []
    for block in content_blocks:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ThinkingBlock):
            text_parts.append(block.thinking)
        elif isinstance(block, RuntimeSnapshotBlock):
            for entry in block.entries:
                text_parts.append(content_blocks_to_text(entry.summary_blocks))
        elif isinstance(block, ToolResultBlock):
            text_parts.append(content_blocks_to_text(block.content_blocks))
    return "\n".join(part for part in text_parts if part)


@dataclass(kw_only=True, init=False)
class Message:
    role: str
    content_blocks: List[ContentBlock]
    raw_content: Optional[str]
    provider_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: Optional[int]

    def __init__(
        self,
        *,
        role: str,
        content: Optional[str] = None,
        content_blocks: Optional[List[ContentBlock]] = None,
        raw_content: Optional[str] = None,
        provider_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        self.role = role
        self.content_blocks = list(content_blocks) if content_blocks is not None else content_blocks_from_text(content or "")
        self.raw_content = raw_content
        self.provider_state = dict(provider_state) if provider_state else None
        self.metadata = dict(metadata) if metadata else {}
        self.timestamp = timestamp

    @property
    def content(self) -> str:
        return content_blocks_to_text(self.content_blocks)

    @content.setter
    def content(self, value: str) -> None:
        self.raw_content = None
        self.content_blocks = content_blocks_from_text(value)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role}
        payload["content"] = self.raw_content if self.raw_content is not None else self.content
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload

    def get_provider_state(self, namespace: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.provider_state:
            return default
        namespaced = self.provider_state.get(namespace)
        if isinstance(namespaced, dict):
            return dict(namespaced)
        return default

    def set_provider_state(self, namespace: str, value: Dict[str, Any]) -> None:
        if self.provider_state is None:
            self.provider_state = {}
        self.provider_state[namespace] = dict(value)


@dataclass(kw_only=True, init=False)
class AssistantMessage(Message):
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]]
    stop_reason: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    api: Optional[str]
    usage: Optional[Dict[str, Any]]

    def __init__(
        self,
        *,
        role: str = "assistant",
        content: Optional[str] = None,
        content_blocks: Optional[List[ContentBlock]] = None,
        raw_content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        stop_reason: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        provider_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        super().__init__(
            role=role,
            content=content,
            content_blocks=content_blocks,
            raw_content=raw_content,
            provider_state=provider_state,
            metadata=metadata,
            timestamp=timestamp,
        )
        self.tool_calls = list(tool_calls) if tool_calls else None
        self.stop_reason = stop_reason
        self.model = model
        self.provider = provider
        self.api = api
        self.usage = dict(usage) if usage else None

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return payload


@dataclass(kw_only=True, init=False)
class ToolResultMessage(Message):
    role: str = "tool"
    tool_call_id: str
    name: str
    details: Any
    is_error: bool

    def __init__(
        self,
        *,
        role: str = "tool",
        tool_call_id: str = "",
        name: str = "",
        content: Optional[str] = None,
        content_blocks: Optional[List[ContentBlock]] = None,
        raw_content: Optional[str] = None,
        details: Any = None,
        is_error: bool = False,
        provider_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[int] = None,
    ):
        super().__init__(
            role=role,
            content=content,
            content_blocks=content_blocks,
            raw_content=raw_content,
            provider_state=provider_state,
            metadata=metadata,
            timestamp=timestamp,
        )
        self.tool_call_id = tool_call_id
        self.name = name
        self.details = details
        self.is_error = is_error

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["tool_call_id"] = self.tool_call_id
        payload["name"] = self.name
        return payload


class CustomMessage:
    """Base class for anything that is not standard LLM Message."""

    role: str = "custom"


AgentMessage = Union[Message, CustomMessage]


@dataclass
class RuntimeVariable:
    key: str
    raw_value: Any
    kind: RuntimeValueKind = "opaque"
    version: int = 1
    updated_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_view: List[ContentBlock] = field(default_factory=list)
    ui_view: List[ContentBlock] = field(default_factory=list)
    dirty: bool = True

    def to_ref(self, *, label: Optional[str] = None) -> "RuntimeRefBlock":
        return RuntimeRefBlock(key=self.key, version=self.version, label=label)


@dataclass
class RuntimeDeltaOp:
    op: RuntimeOperationType
    key: str
    value: Any = None
    path: Optional[List[Union[str, int]]] = None
    blocks: List[ContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeState:
    variables: Dict[str, RuntimeVariable] = field(default_factory=dict)
    revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_variable(
        self,
        key: str,
        value: Any,
        *,
        kind: RuntimeValueKind = "opaque",
        updated_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_view: Optional[List[ContentBlock]] = None,
        ui_view: Optional[List[ContentBlock]] = None,
    ) -> RuntimeVariable:
        current = self.variables.get(key)
        version = (current.version + 1) if current else 1
        variable = RuntimeVariable(
            key=key,
            raw_value=value,
            kind=kind,
            version=version,
            updated_by=updated_by,
            metadata=dict(metadata) if metadata else (dict(current.metadata) if current else {}),
            llm_view=list(llm_view) if llm_view is not None else (list(current.llm_view) if current else []),
            ui_view=list(ui_view) if ui_view is not None else (list(current.ui_view) if current else []),
            dirty=True,
        )
        self.variables[key] = variable
        self.revision += 1
        return variable

    def delete_variable(self, key: str) -> None:
        if key in self.variables:
            del self.variables[key]
            self.revision += 1

    def apply_op(self, op: RuntimeDeltaOp, *, updated_by: Optional[str] = None) -> None:
        variable = self.variables.get(op.key)

        if op.op == "delete":
            self.delete_variable(op.key)
            return

        if op.op == "touch":
            if variable is not None:
                variable.version += 1
                variable.updated_by = updated_by
                variable.dirty = True
                if op.metadata:
                    variable.metadata.update(op.metadata)
                self.revision += 1
            return

        if op.op == "merge":
            current = variable.raw_value if variable else {}
            merged = dict(current) if isinstance(current, dict) else {}
            if isinstance(op.value, dict):
                merged.update(op.value)
            self.set_variable(
                op.key,
                merged,
                kind="structured",
                updated_by=updated_by,
                metadata=op.metadata or (variable.metadata if variable else None),
                llm_view=op.blocks or (variable.llm_view if variable else None),
                ui_view=op.blocks or (variable.ui_view if variable else None),
            )
            return

        if op.op == "append":
            current = list(variable.raw_value) if variable and isinstance(variable.raw_value, list) else []
            if isinstance(op.value, list):
                current.extend(op.value)
            else:
                current.append(op.value)
            self.set_variable(
                op.key,
                current,
                kind="structured",
                updated_by=updated_by,
                metadata=op.metadata or (variable.metadata if variable else None),
                llm_view=op.blocks or (variable.llm_view if variable else None),
                ui_view=op.blocks or (variable.ui_view if variable else None),
            )
            return

        if op.op == "replace_blocks":
            raw_value = variable.raw_value if variable is not None else op.value
            self.set_variable(
                op.key,
                raw_value,
                kind=variable.kind if variable is not None else "message_blocks",
                updated_by=updated_by,
                metadata=op.metadata or (variable.metadata if variable else None),
                llm_view=list(op.blocks),
                ui_view=list(op.blocks),
            )
            return

        self.set_variable(
            op.key,
            op.value,
            kind=self._infer_value_kind(op.value),
            updated_by=updated_by,
            metadata=op.metadata or (variable.metadata if variable else None),
            llm_view=list(op.blocks) if op.blocks else (variable.llm_view if variable else None),
            ui_view=list(op.blocks) if op.blocks else (variable.ui_view if variable else None),
        )

    def apply_ops(self, ops: List[RuntimeDeltaOp], *, updated_by: Optional[str] = None) -> None:
        for op in ops:
            self.apply_op(op, updated_by=updated_by)

    def get_raw_memory(self) -> Dict[str, Any]:
        return {key: variable.raw_value for key, variable in self.variables.items()}

    def sync_from_shared_memory(self, shared_memory: Dict[str, Any]) -> None:
        for key, value in shared_memory.items():
            if key not in self.variables:
                self.set_variable(key, value)

    def sync_to_shared_memory(self, shared_memory: Dict[str, Any]) -> None:
        shared_memory.clear()
        shared_memory.update(self.get_raw_memory())

    @staticmethod
    def _infer_value_kind(value: Any) -> RuntimeValueKind:
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - optional dependency path
            pd = None

        if pd is not None and isinstance(value, pd.DataFrame):
            return "tabular"
        if isinstance(value, list) and value and all(isinstance(item, BaseContentBlock) for item in value):
            return "message_blocks"
        if isinstance(value, (dict, list, tuple)):
            return "structured"
        return "opaque"


@dataclass
class AgentContext:
    """Context passed to tools during execution."""

    messages: List[AgentMessage]
    runtime: RuntimeState = field(default_factory=RuntimeState)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    active_tool_selection: Optional["ToolRuntimeSelection"] = None

    def __post_init__(self) -> None:
        if self.shared_memory:
            self.runtime.sync_from_shared_memory(self.shared_memory)
        else:
            self.shared_memory.update(self.runtime.get_raw_memory())

    def refresh_runtime_from_shared_memory(self) -> None:
        self.runtime.sync_from_shared_memory(self.shared_memory)

    def refresh_shared_memory_from_runtime(self) -> None:
        self.runtime.sync_to_shared_memory(self.shared_memory)

    def get_runtime_variable(self, key: str) -> Optional[RuntimeVariable]:
        return self.runtime.variables.get(key)

    def select_runtime_variables(self, keys: Optional[List[str]] = None) -> Dict[str, RuntimeVariable]:
        if keys is None:
            return dict(self.runtime.variables)
        return {key: variable for key, variable in self.runtime.variables.items() if key in keys}

    def build_tool_runtime_selection(
        self,
        tool_name: str,
        *,
        reads: Optional[List[str]] = None,
        writes: Optional[List[str]] = None,
        temp_outputs: Optional[List[str]] = None,
    ) -> "ToolRuntimeSelection":
        requested_reads = list(reads or [])
        selected_inputs = self.select_runtime_variables(requested_reads)
        missing_reads = [key for key in requested_reads if key not in selected_inputs]
        return ToolRuntimeSelection(
            tool_name=tool_name,
            reads=requested_reads,
            writes=list(writes or []),
            temp_outputs=list(temp_outputs or []),
            input_variables=selected_inputs,
            missing_reads=missing_reads,
        )


@dataclass
class ToolRuntimeSelection:
    tool_name: str
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    temp_outputs: List[str] = field(default_factory=list)
    input_variables: Dict[str, RuntimeVariable] = field(default_factory=dict)
    missing_reads: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "temp_outputs": list(self.temp_outputs),
            "input_variables": {
                key: {
                    "key": variable.key,
                    "version": variable.version,
                    "kind": variable.kind,
                    "updated_by": variable.updated_by,
                    "metadata": dict(variable.metadata),
                }
                for key, variable in self.input_variables.items()
            },
            "missing_reads": list(self.missing_reads),
        }


@dataclass(init=False)
class ToolResult:
    """Structured result returned by a tool."""

    content_blocks: List[ContentBlock] = field(default_factory=list)
    raw_content: Optional[str] = None
    details: Any = None
    state_delta: Optional[Dict[str, Any]] = None
    runtime_ops: List[RuntimeDeltaOp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        content: Optional[str] = None,
        *,
        content_blocks: Optional[List[ContentBlock]] = None,
        raw_content: Optional[str] = None,
        details: Any = None,
        state_delta: Optional[Dict[str, Any]] = None,
        runtime_ops: Optional[List[RuntimeDeltaOp]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content_blocks = list(content_blocks) if content_blocks is not None else content_blocks_from_text(content or "")
        self.raw_content = raw_content
        self.details = details
        self.state_delta = state_delta
        self.runtime_ops = list(runtime_ops) if runtime_ops else []
        self.metadata = dict(metadata) if metadata else {}

    @property
    def content(self) -> str:
        if self.raw_content is not None:
            return self.raw_content
        return content_blocks_to_text(self.content_blocks)

    @content.setter
    def content(self, value: str) -> None:
        self.raw_content = None
        self.content_blocks = content_blocks_from_text(value)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        state_delta: Optional[Dict[str, Any]] = None,
        runtime_ops: Optional[List[RuntimeDeltaOp]] = None,
        details: Any = None,
    ) -> "ToolResult":
        return cls(
            content_blocks=content_blocks_from_text(text),
            state_delta=state_delta,
            runtime_ops=list(runtime_ops) if runtime_ops else [],
            details=details,
        )


@dataclass
class AgentToolResult:
    """Legacy tool update/result payload kept for compatibility."""

    content: str = ""
    details: Any = None


@dataclass(init=False)
class AgentToolUpdate:
    content_blocks: List[ContentBlock] = field(default_factory=list)
    raw_content: Optional[str] = None
    details: Any = None
    runtime_ops: List[RuntimeDeltaOp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        content: Optional[str] = None,
        *,
        content_blocks: Optional[List[ContentBlock]] = None,
        raw_content: Optional[str] = None,
        details: Any = None,
        runtime_ops: Optional[List[RuntimeDeltaOp]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content_blocks = list(content_blocks) if content_blocks is not None else content_blocks_from_text(content or "")
        self.raw_content = raw_content
        self.details = details
        self.runtime_ops = list(runtime_ops) if runtime_ops else []
        self.metadata = dict(metadata) if metadata else {}

    @property
    def content(self) -> str:
        if self.raw_content is not None:
            return self.raw_content
        return content_blocks_to_text(self.content_blocks)


class StateReducer(Protocol):
    """Protocol for reducing large objects into LLM-friendly summaries."""

    def reduce(self, obj: Any) -> Union[str, List[ContentBlock]]:
        ...


class EnvironmentInspector(Protocol):
    """Protocol for capturing runtime state/metadata after tool executions."""

    async def capture_state(self, context: AgentContext) -> Union[str, List[ContentBlock], RuntimeSnapshotBlock]:
        ...


class CancellationSignal(Protocol):
    is_cancelled: bool
    reason: Optional[str]

    def throw_if_cancelled(self) -> None:
        ...


@dataclass
class BasicCancellationSignal:
    is_cancelled: bool = False
    reason: Optional[str] = None

    def cancel(self, reason: str = "Operation cancelled") -> None:
        self.is_cancelled = True
        self.reason = reason

    def throw_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RuntimeError(self.reason or "Operation cancelled")


class AgentToolUpdateCallback(Protocol):
    def __call__(self, partial_result: AgentToolUpdate) -> None: ...


class AgentTool:
    """Python equivalent of AgentTool with an execute function."""

    name: str
    description: str
    parameters: Dict[str, Any]
    label: str
    reads: List[str]
    writes: List[str]
    temp_outputs: List[str]

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        label: str,
        *,
        reads: Optional[List[str]] = None,
        writes: Optional[List[str]] = None,
        temp_outputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.label = label
        self.reads = list(reads) if reads else []
        self.writes = list(writes) if writes else []
        self.temp_outputs = list(temp_outputs) if temp_outputs else []

    def resolve_runtime_selection(
        self,
        context: AgentContext,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolRuntimeSelection:
        return context.build_tool_runtime_selection(
            self.name,
            reads=self.reads,
            writes=self.writes,
            temp_outputs=self.temp_outputs,
        )

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        context: AgentContext,
        on_update: Optional[AgentToolUpdateCallback] = None,
        signal: Optional[CancellationSignal] = None,
    ) -> ToolResult:
        raise NotImplementedError


class AgentLoopConfig(Protocol):
    """Configuration for the agent loop."""

    max_rounds: Optional[int]
    max_consecutive_tool_failures: Optional[int]
    model: Optional[Any]
    thinking_level: Optional[str]
    system_prompt: Optional[str]
    session_id: Optional[str]
    transport: Optional[str]
    thinking_budgets: Optional[Dict[str, Any]]
    on_payload: Optional[Any]

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        ...

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        ...

    async def compact_messages(
        self,
        messages: List[AgentMessage],
        context: AgentContext,
    ) -> List[AgentMessage]:
        ...

    async def get_steering_messages(self) -> List[AgentMessage]:
        ...

    async def get_followup_messages(self) -> List[AgentMessage]:
        ...

    async def handle_consolidation(
        self,
        messages: List[AgentMessage],
        context: AgentContext,
    ) -> Optional[List[AgentMessage]]:
        ...


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Any = "gpt-4"
    thinking_level: str = "off"
    tools: List[AgentTool] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    runtime: RuntimeState = field(default_factory=RuntimeState)
    is_streaming: bool = False
    stream_message: Optional[AgentMessage] = None
    pending_tool_calls: set = field(default_factory=set)
    error: Optional[str] = None
    session_id: Optional[str] = None
    transport: Optional[str] = None
    thinking_budgets: Optional[Dict[str, Any]] = None
    on_payload: Optional[Any] = None


@dataclass
class AgentEvent:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    parent_id: Optional[str] = None
