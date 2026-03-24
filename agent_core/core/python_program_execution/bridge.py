from __future__ import annotations

import hashlib
import inspect
import json
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ...assistant_messages.runtime import AgentContext, RuntimeDeltaOp
from .types import PythonProgramBlock, PythonProgramExecutionRequest, PythonProgramExecutionResult


@dataclass
class PythonRuntimeBridgeResult:
    runtime_ops: List[RuntimeDeltaOp] = field(default_factory=list)
    synced_variables: List[str] = field(default_factory=list)
    skipped_variables: Dict[str, str] = field(default_factory=dict)
    namespace_before_keys: List[str] = field(default_factory=list)
    namespace_after_keys: List[str] = field(default_factory=list)


class PythonRuntimeBridge(Protocol):
    async def build_runtime_bridge_result(
        self,
        *,
        execution_request: PythonProgramExecutionRequest,
        execution_result: PythonProgramExecutionResult,
        agent_context: AgentContext,
        python_block: PythonProgramBlock,
    ) -> PythonRuntimeBridgeResult:
        ...


class NamespaceValueProjector(Protocol):
    def supports(self, name: str, value: Any) -> bool:
        ...

    def fingerprint(self, value: Any) -> Optional[str]:
        ...

    def build_runtime_op(self, name: str, value: Any, fingerprint: Optional[str]) -> RuntimeDeltaOp:
        ...


def _stable_digest(value: Any) -> str:
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()


def _is_pandas_dataframe(value: Any) -> bool:
    type_repr = str(type(value))
    if type_repr == "<class 'pandas.core.frame.DataFrame'>":
        return True
    return (
        hasattr(value, "columns")
        and hasattr(value, "dtypes")
        and hasattr(value, "head")
        and hasattr(value, "shape")
        and hasattr(value, "to_json")
    )


def _is_pandas_series(value: Any) -> bool:
    type_repr = str(type(value))
    if type_repr == "<class 'pandas.core.series.Series'>":
        return True
    return (
        hasattr(value, "index")
        and hasattr(value, "dtype")
        and hasattr(value, "head")
        and hasattr(value, "to_dict")
        and not hasattr(value, "columns")
    )


class ScalarProjector:
    def supports(self, name: str, value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    def fingerprint(self, value: Any) -> Optional[str]:
        return _stable_digest(value)

    def build_runtime_op(self, name: str, value: Any, fingerprint: Optional[str]) -> RuntimeDeltaOp:
        return RuntimeDeltaOp(
            op="set",
            key=name,
            value=value,
            metadata={
                "python_bridge": True,
                "python_variable": name,
                "python_variable_kind": "scalar",
                "python_bridge_fingerprint": fingerprint,
            },
        )


class StructuredProjector:
    def supports(self, name: str, value: Any) -> bool:
        return isinstance(value, (dict, list, tuple))

    def fingerprint(self, value: Any) -> Optional[str]:
        try:
            normalized = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            normalized = repr(value)
        return _stable_digest(normalized)

    def build_runtime_op(self, name: str, value: Any, fingerprint: Optional[str]) -> RuntimeDeltaOp:
        metadata: Dict[str, Any] = {
            "python_bridge": True,
            "python_variable": name,
            "python_variable_kind": "structured",
            "python_bridge_fingerprint": fingerprint,
        }
        if isinstance(value, dict):
            metadata["key_count"] = len(value)
        if isinstance(value, (list, tuple)):
            metadata["length"] = len(value)
        return RuntimeDeltaOp(op="set", key=name, value=value, metadata=metadata)


class DataFrameProjector:
    def supports(self, name: str, value: Any) -> bool:
        return _is_pandas_dataframe(value)

    def fingerprint(self, value: Any) -> Optional[str]:
        try:
            shape = tuple(value.shape)
            columns = list(value.columns)
            dtypes = [str(dtype) for dtype in value.dtypes]
            preview = value.head(5).to_json(orient="split", date_format="iso", default_handler=str)
            return _stable_digest((shape, columns, dtypes, preview))
        except Exception:
            return _stable_digest(repr(value))

    def build_runtime_op(self, name: str, value: Any, fingerprint: Optional[str]) -> RuntimeDeltaOp:
        shape = getattr(value, "shape", None)
        columns = list(getattr(value, "columns", []))
        metadata: Dict[str, Any] = {
            "python_bridge": True,
            "python_variable": name,
            "python_variable_kind": "tabular",
            "python_bridge_fingerprint": fingerprint,
        }
        if isinstance(shape, tuple) and len(shape) == 2:
            metadata["rows"] = shape[0]
            metadata["columns_count"] = shape[1]
        metadata["columns"] = columns[:20]
        return RuntimeDeltaOp(op="set", key=name, value=value, metadata=metadata)


class SeriesProjector:
    def supports(self, name: str, value: Any) -> bool:
        return _is_pandas_series(value)

    def fingerprint(self, value: Any) -> Optional[str]:
        try:
            preview = value.head(10).to_json(date_format="iso", default_handler=str)
            dtype = str(value.dtype)
            index_preview = list(value.index[:10])
            return _stable_digest((dtype, index_preview, preview))
        except Exception:
            return _stable_digest(repr(value))

    def build_runtime_op(self, name: str, value: Any, fingerprint: Optional[str]) -> RuntimeDeltaOp:
        metadata: Dict[str, Any] = {
            "python_bridge": True,
            "python_variable": name,
            "python_variable_kind": "series",
            "python_bridge_fingerprint": fingerprint,
            "length": int(len(value)) if hasattr(value, "__len__") else None,
            "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
        }
        if hasattr(value, "name") and value.name is not None:
            metadata["series_name"] = str(value.name)
        return RuntimeDeltaOp(op="set", key=name, value=value, metadata=metadata)


class DefaultPythonRuntimeBridge:
    def __init__(
        self,
        *,
        projectors: Optional[Sequence[NamespaceValueProjector]] = None,
        ignored_names: Optional[Sequence[str]] = None,
    ):
        self.projectors = list(projectors) if projectors is not None else [
            DataFrameProjector(),
            SeriesProjector(),
            ScalarProjector(),
            StructuredProjector(),
        ]
        self.ignored_names = set(ignored_names or []) | {
            "__builtins__",
            "In",
            "Out",
            "get_ipython",
            "exit",
            "quit",
            "runtime",
            "python_runtime",
            "shared_memory",
            "messages",
            "assistant_message",
            "python_block",
        }

    async def build_runtime_bridge_result(
        self,
        *,
        execution_request: PythonProgramExecutionRequest,
        execution_result: PythonProgramExecutionResult,
        agent_context: AgentContext,
        python_block: PythonProgramBlock,
    ) -> PythonRuntimeBridgeResult:
        del agent_context
        del python_block

        result = PythonRuntimeBridgeResult()
        before_namespace = dict(execution_request.namespace)
        after_namespace = dict(execution_result.namespace)
        result.namespace_before_keys = sorted(before_namespace.keys())
        result.namespace_after_keys = sorted(after_namespace.keys())

        for name, value in after_namespace.items():
            if not self._should_consider_name(name):
                result.skipped_variables[name] = "ignored_name"
                continue
            if self._is_runtime_internal(value):
                result.skipped_variables[name] = "runtime_internal"
                continue

            projector = self._find_projector(name, value)
            if projector is None:
                result.skipped_variables[name] = "unsupported_type"
                continue

            fingerprint = projector.fingerprint(value)
            existing_before = before_namespace.get(name)
            if name in before_namespace and self._is_unchanged(existing_before, value, fingerprint):
                result.skipped_variables[name] = "unchanged"
                continue

            runtime_op = projector.build_runtime_op(name, value, fingerprint)
            result.runtime_ops.append(runtime_op)
            result.synced_variables.append(name)

        return result

    def _should_consider_name(self, name: str) -> bool:
        if name in self.ignored_names:
            return False
        if not name:
            return False
        if name.startswith("_"):
            return False
        return True

    def _is_runtime_internal(self, value: Any) -> bool:
        if isinstance(value, types.ModuleType):
            return True
        if inspect.isclass(value):
            return True
        if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
            return True
        if callable(value):
            return True
        return False

    def _find_projector(self, name: str, value: Any) -> Optional[NamespaceValueProjector]:
        for projector in self.projectors:
            if projector.supports(name, value):
                return projector
        return None

    def _is_unchanged(self, previous_value: Any, current_value: Any, fingerprint: Optional[str]) -> bool:
        if previous_value is current_value and fingerprint is None:
            return True
        projector = self._find_projector("", current_value)
        if projector is None:
            return False
        previous_fingerprint = projector.fingerprint(previous_value) if projector.supports("", previous_value) else None
        if fingerprint is not None and previous_fingerprint is not None:
            return fingerprint == previous_fingerprint
        try:
            return previous_value == current_value
        except Exception:
            return False


class NullPythonRuntimeBridge:
    async def build_runtime_bridge_result(
        self,
        *,
        execution_request: PythonProgramExecutionRequest,
        execution_result: PythonProgramExecutionResult,
        agent_context: AgentContext,
        python_block: PythonProgramBlock,
    ) -> PythonRuntimeBridgeResult:
        del execution_request
        del execution_result
        del agent_context
        del python_block
        return PythonRuntimeBridgeResult()


def resolve_python_runtime_bridge(config: Any) -> PythonRuntimeBridge:
    configured = getattr(config, "python_runtime_bridge", None)
    if configured is False:
        return NullPythonRuntimeBridge()
    if configured is None:
        return DefaultPythonRuntimeBridge()
    if hasattr(configured, "build_runtime_bridge_result"):
        return configured
    return DefaultPythonRuntimeBridge()


__all__ = [
    "DataFrameProjector",
    "DefaultPythonRuntimeBridge",
    "NamespaceValueProjector",
    "NullPythonRuntimeBridge",
    "PythonRuntimeBridge",
    "PythonRuntimeBridgeResult",
    "ScalarProjector",
    "SeriesProjector",
    "StructuredProjector",
    "resolve_python_runtime_bridge",
]
