from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .messages import AgentMessage, Message
from .runtime import AgentContext, RuntimeState
from .tools import AgentTool


class AgentLoopConfig(Protocol):
    api_key: Optional[str]
    max_rounds: Optional[int]
    max_consecutive_tool_failures: Optional[int]
    model: Optional[Any]
    thinking_level: Optional[str]
    system_prompt: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    session_id: Optional[str]
    transport: Optional[str]
    cache_retention: Optional[str]
    headers: Optional[Dict[str, str]]
    max_retry_delay_ms: Optional[int]
    metadata: Optional[Dict[str, Any]]
    thinking_budgets: Optional[Dict[str, Any]]
    on_payload: Optional[Any]

    async def get_api_key(self, provider: str) -> Optional[str]:
        ...

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
class AssistantDelta:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    provider_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    usage: Optional[Dict[str, Any]] = None
    raw_content: Optional[str] = None

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any]) -> "AssistantDelta":
        provider_state: Dict[str, Dict[str, Any]] = {}
        raw_provider_state = chunk.get("provider_state")
        if isinstance(raw_provider_state, dict):
            for namespace, payload in raw_provider_state.items():
                if isinstance(namespace, str) and isinstance(payload, dict):
                    provider_state[namespace] = dict(payload)

        raw_tool_calls = chunk.get("tool_calls")
        tool_calls = [dict(tool_call) for tool_call in raw_tool_calls] if isinstance(raw_tool_calls, list) else []

        usage = chunk.get("usage")
        normalized_usage = dict(usage) if isinstance(usage, dict) else None

        content = chunk.get("content")
        reasoning = chunk.get("reasoning")
        raw_content = chunk.get("raw_content")

        return cls(
            content=str(content) if content is not None else None,
            reasoning=str(reasoning) if reasoning is not None else None,
            tool_calls=tool_calls,
            provider_state=provider_state,
            usage=normalized_usage,
            raw_content=str(raw_content) if raw_content is not None else None,
        )

    def to_chunk(self) -> Dict[str, Any]:
        chunk: Dict[str, Any] = {}
        if self.content is not None:
            chunk["content"] = self.content
        if self.reasoning is not None:
            chunk["reasoning"] = self.reasoning
        if self.tool_calls:
            chunk["tool_calls"] = [dict(tool_call) for tool_call in self.tool_calls]
        if self.provider_state:
            chunk["provider_state"] = {
                namespace: dict(payload) for namespace, payload in self.provider_state.items()
            }
        if self.usage is not None:
            chunk["usage"] = dict(self.usage)
        if self.raw_content is not None:
            chunk["raw_content"] = self.raw_content
        return chunk

    @property
    def has_text(self) -> bool:
        return bool(self.content)

    @property
    def has_reasoning(self) -> bool:
        return bool(self.reasoning)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Any = "gpt-4"
    thinking_level: str = "off"
    tools: List[AgentTool] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    runtime: RuntimeState = field(default_factory=RuntimeState)
    python_runtime: Optional[Any] = None
    host_context: Optional[Any] = None
    is_streaming: bool = False
    stream_message: Optional[AgentMessage] = None
    pending_tool_calls: set = field(default_factory=set)
    error: Optional[str] = None
    session_id: Optional[str] = None
    transport: Optional[str] = None
    cache_retention: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    max_retry_delay_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    thinking_budgets: Optional[Dict[str, Any]] = None
    on_payload: Optional[Any] = None


@dataclass
class AgentEvent:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    parent_id: Optional[str] = None

    @property
    def payload(self) -> Dict[str, Any]:
        return self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def message(self) -> Any:
        return self.data.get("message")

    @property
    def messages(self) -> Any:
        return self.data.get("messages")

    @property
    def tool_results(self) -> Any:
        return self.data.get("tool_results")

    @property
    def tool_call(self) -> Any:
        return self.data.get("tool_call")

    @property
    def tool_call_id(self) -> Optional[str]:
        direct = self.data.get("tool_call_id")
        if direct is not None:
            return direct
        tool_call = self.tool_call
        return getattr(tool_call, "id", None)

    @property
    def tool_name(self) -> Optional[str]:
        direct = self.data.get("tool_name")
        if direct is not None:
            return direct
        tool_call = self.tool_call
        return getattr(tool_call, "name", None)

    @property
    def args(self) -> Any:
        if "args" in self.data:
            return self.data.get("args")
        tool_call = self.tool_call
        return getattr(tool_call, "arguments", None)

    @property
    def partial_result(self) -> Any:
        return self.data.get("partial_result")

    @property
    def result(self) -> Any:
        return self.data.get("result")

    @property
    def error(self) -> Any:
        return self.data.get("error")

    @property
    def delta(self) -> Any:
        return self.data.get("delta")

    @property
    def assistant_delta(self) -> Optional[AssistantDelta]:
        delta = self.data.get("delta")
        if isinstance(delta, AssistantDelta):
            return delta
        return None

    @property
    def round(self) -> Optional[int]:
        return self.data.get("round")

    @property
    def runtime_revision(self) -> Optional[int]:
        return self.data.get("runtime_revision")

    @property
    def steering_messages(self) -> Any:
        return self.data.get("steering_messages")

    @property
    def is_error(self) -> Optional[bool]:
        if "is_error" in self.data:
            return bool(self.data["is_error"])
        return None


__all__ = [
    "AssistantDelta",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentState",
]
