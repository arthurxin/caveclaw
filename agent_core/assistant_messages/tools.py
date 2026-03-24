from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

from .content_blocks import ContentBlock, RuntimeSnapshotBlock, content_blocks_from_text, content_blocks_to_text
from .runtime import AgentContext, RuntimeDeltaOp, ToolRuntimeSelection


@dataclass(init=False)
class ToolResult:
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
    def reduce(self, obj: Any) -> Union[str, List[ContentBlock]]:
        ...


class EnvironmentInspector(Protocol):
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
    def __call__(self, partial_result: AgentToolUpdate) -> None:
        ...


class AgentTool:
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


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdate",
    "AgentToolUpdateCallback",
    "BasicCancellationSignal",
    "CancellationSignal",
    "EnvironmentInspector",
    "StateReducer",
    "ToolResult",
]
