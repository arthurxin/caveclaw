from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Union

from ..assistant_messages.types import (
    AgentContext,
    ContentBlock,
    EnvironmentInspector,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    RuntimeVariable,
    StateReducer,
    TextBlock,
    content_blocks_from_text,
)


class DataFrameReducer(StateReducer):
    """Summarize a DataFrame without forcing pandas as a hard dependency."""

    def reduce(self, obj: Any) -> Union[str, List[ContentBlock]]:
        if "pandas" in sys.modules and str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
            try:
                shape = obj.shape
                cols = list(obj.columns)
                head_str = obj.head(3).to_string()
                memory_usage = obj.memory_usage(deep=True).sum() / (1024 * 1024)
                return (
                    f"DataFrame:\n"
                    f"  - Shape: {shape[0]} rows x {shape[1]} cols\n"
                    f"  - Memory: ~{memory_usage:.2f} MB\n"
                    f"  - Columns: {cols}\n"
                    f"  - Head (3 rows):\n{head_str}"
                )
            except Exception as error:  # pragma: no cover - defensive branch
                return f"[Error reducing DataFrame: {error}]"
        return str(obj)


class ListReducer(StateReducer):
    """Summarize long lists to keep runtime snapshots compact."""

    def __init__(self, max_items: int = 5):
        self.max_items = max_items

    def reduce(self, obj: Any) -> Union[str, List[ContentBlock]]:
        if isinstance(obj, list):
            if len(obj) <= self.max_items:
                return str(obj)
            head = obj[:self.max_items]
            return f"List (len={len(obj)}): {str(head)[:-1]}, ... <{len(obj) - self.max_items} more items>]"
        return str(obj)


class PythonRuntimeInspector(EnvironmentInspector):
    """Runtime-first inspector that projects `RuntimeState` into snapshot blocks."""

    def __init__(self, reducers: Optional[List[StateReducer]] = None):
        self.reducers = reducers or [DataFrameReducer(), ListReducer()]

    def _reduce_value(self, value: Any) -> List[ContentBlock]:
        reduced_value: Optional[Union[str, List[ContentBlock]]] = None

        for reducer in self.reducers:
            reduced_attempt = reducer.reduce(value)
            if isinstance(reduced_attempt, list):
                return list(reduced_attempt)
            if reduced_attempt != str(value):
                reduced_value = reduced_attempt
                break

        if reduced_value is None:
            if isinstance(value, dict):
                keys = list(value.keys())
                if len(keys) > 10:
                    reduced_value = f"Dict with {len(keys)} keys: {keys[:10]}..."
                else:
                    reduced_value = str(value)
            else:
                raw_str = str(value)
                if len(raw_str) > 500:
                    reduced_value = raw_str[:500] + f"... <Truncated {len(raw_str) - 500} chars>"
                else:
                    reduced_value = raw_str

        return content_blocks_from_text(str(reduced_value))

    def _build_entry(self, variable: RuntimeVariable) -> RuntimeSnapshotEntry:
        summary_blocks = list(variable.llm_view) if variable.llm_view else self._reduce_value(variable.raw_value)
        visible_metadata = {
            key: value
            for key, value in variable.metadata.items()
            if (isinstance(value, (str, int, float, bool)) or value is None) and not key.startswith("ui_")
        }
        metadata: Dict[str, Any] = {
            "kind": variable.kind,
            "updated_by": variable.updated_by,
            **visible_metadata,
        }
        return RuntimeSnapshotEntry(
            key=variable.key,
            version=variable.version,
            summary_blocks=summary_blocks,
            metadata=metadata,
        )

    async def capture_state(self, context: AgentContext) -> RuntimeSnapshotBlock:
        runtime = context.runtime
        if not runtime.variables and context.shared_memory:
            context.refresh_runtime_from_shared_memory()

        if not runtime.variables:
            return RuntimeSnapshotBlock(
                entries=[
                    RuntimeSnapshotEntry(
                        key="runtime",
                        version=runtime.revision,
                        summary_blocks=[TextBlock(text="Environment State: [Memory empty or unchanged]")],
                    )
                ]
            )

        entries = [self._build_entry(variable) for variable in runtime.variables.values()]
        return RuntimeSnapshotBlock(entries=entries, metadata={"runtime_revision": runtime.revision})
