from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .content_blocks import ContentBlock, ToolCall, content_blocks_from_text, content_blocks_to_text


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
    role: str = "custom"


AgentMessage = Union[Message, CustomMessage]


__all__ = [
    "AgentMessage",
    "AssistantMessage",
    "CustomMessage",
    "Message",
    "ToolResultMessage",
]
