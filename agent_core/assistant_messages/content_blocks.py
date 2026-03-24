from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


ContentBlockType = Literal[
    "text",
    "image",
    "thinking",
    "tool_call",
    "tool_result",
    "runtime_ref",
    "runtime_snapshot",
]


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
    segments: List[tuple[str, str]] = []

    def append_segment(kind: str, text: str) -> None:
        if not text:
            return
        if segments and segments[-1][0] == kind and kind in {"text", "thinking"}:
            prev_kind, prev_text = segments[-1]
            segments[-1] = (prev_kind, prev_text + text)
        else:
            segments.append((kind, text))

    for block in content_blocks:
        if isinstance(block, TextBlock):
            append_segment("text", block.text)
        elif isinstance(block, ThinkingBlock):
            append_segment("thinking", block.thinking)
        elif isinstance(block, RuntimeSnapshotBlock):
            snapshot_parts = [content_blocks_to_text(entry.summary_blocks) for entry in block.entries]
            snapshot_text = "\n".join(part for part in snapshot_parts if part)
            if snapshot_text:
                append_segment("runtime_snapshot", snapshot_text)
        elif isinstance(block, ToolResultBlock):
            nested_text = content_blocks_to_text(block.content_blocks)
            if nested_text:
                append_segment("tool_result", nested_text)
    return "\n".join(text for _, text in segments if text)


__all__ = [
    "BaseContentBlock",
    "ContentBlock",
    "ContentBlockType",
    "ImageBlock",
    "RuntimeRefBlock",
    "RuntimeSnapshotBlock",
    "RuntimeSnapshotEntry",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolCallBlock",
    "ToolResultBlock",
    "content_blocks_from_text",
    "content_blocks_to_text",
    "text_block",
]
