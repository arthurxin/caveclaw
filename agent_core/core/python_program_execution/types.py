from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


PythonProgramBackend = Literal["ipython", "python"]


@dataclass
class PythonProgramBlock:
    code: str
    language: str = "python"
    raw_fence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PythonProgramExecutionRequest:
    block: PythonProgramBlock
    backend: PythonProgramBackend = "ipython"
    namespace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PythonProgramExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    backend: PythonProgramBackend = "ipython"
    namespace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PythonProgramBackend",
    "PythonProgramBlock",
    "PythonProgramExecutionRequest",
    "PythonProgramExecutionResult",
]
