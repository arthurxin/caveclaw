from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from .content_blocks import BaseContentBlock, ContentBlock, RuntimeRefBlock, RuntimeSnapshotBlock
from .messages import AgentMessage


RuntimeValueKind = Literal["opaque", "structured", "tabular", "message_blocks", "binary_ref"]
RuntimeOperationType = Literal["set", "delete", "merge", "append", "touch", "replace_blocks"]


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

    def to_ref(self, *, label: Optional[str] = None) -> RuntimeRefBlock:
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
        except Exception:  # pragma: no cover
            pd = None

        if pd is not None and isinstance(value, pd.DataFrame):
            return "tabular"
        if isinstance(value, list) and value and all(isinstance(item, BaseContentBlock) for item in value):
            return "message_blocks"
        if isinstance(value, (dict, list, tuple)):
            return "structured"
        return "opaque"


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


@dataclass
class AgentContext:
    messages: List[AgentMessage]
    runtime: RuntimeState = field(default_factory=RuntimeState)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    active_tool_selection: Optional[ToolRuntimeSelection] = None
    python_runtime: Optional[Any] = None

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
    ) -> ToolRuntimeSelection:
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


__all__ = [
    "AgentContext",
    "RuntimeDeltaOp",
    "RuntimeOperationType",
    "RuntimeState",
    "RuntimeValueKind",
    "RuntimeVariable",
    "ToolRuntimeSelection",
]
