from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..assistant_messages.types import AgentMessage, AgentTool, RuntimeState
from .python_program_execution import IPythonProgramRuntime


ApiKeyResolver = Callable[[str], Optional[str] | Awaitable[Optional[str]]]


@dataclass
class AgentHostContext:
    runtime: RuntimeState = field(default_factory=RuntimeState)
    python_runtime: Any = field(default_factory=IPythonProgramRuntime)
    model_registry: Optional[Any] = None
    api_key: Optional[str] = None
    get_api_key: Optional[ApiKeyResolver] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSessionContext:
    messages: List[AgentMessage] = field(default_factory=list)
    tools: List[AgentTool] = field(default_factory=list)
    host: AgentHostContext = field(default_factory=AgentHostContext)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentHostContext",
    "AgentSessionContext",
    "ApiKeyResolver",
]
