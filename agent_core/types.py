from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Protocol, AsyncGenerator, Set


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.tool_calls:
            d["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}} for tc in self.tool_calls]
        return d


@dataclass
class ToolResultMessage(Message):
    role: str = "tool"
    tool_call_id: str = ""
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["tool_call_id"] = self.tool_call_id
        d["name"] = self.name
        return d

# AgentMessage represents unified message representation (LLM messages + Custom UI messages)
class CustomMessage:
    """Base class for anything that is not standard LLM Message."""
    pass

AgentMessage = Union[Message, CustomMessage]


@dataclass
class AgentToolResult:
    content: str
    details: Any = None


class AgentToolUpdateCallback(Protocol):
    def __call__(self, partial_result: Any) -> None: ...


class AgentTool:
    """Python equivalent of AgentTool with an execute function."""
    name: str
    description: str
    parameters: Dict[str, Any]
    label: str

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], label: str):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.label = label

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        on_update: Optional[AgentToolUpdateCallback] = None
    ) -> AgentToolResult:
        raise NotImplementedError


class AgentLoopConfig(Protocol):
    """Configuration for the agent loop."""

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        """Convert mixed messages to LLM-compatible format."""
        ...

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """Optional hook to run contextual compaction before sending to LLM."""
        ...

    async def get_steering_messages(self) -> List[AgentMessage]:
        """Called mid-generation/tool-execution to check for interruptions."""
        ...

    async def get_followup_messages(self) -> List[AgentMessage]:
        """Called when no tools are pending."""
        ...


@dataclass
class AgentState:
    system_prompt: str = ""
    model: str = "gpt-4"
    thinking_level: str = "off"
    tools: List[AgentTool] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    pending_tool_calls: set = field(default_factory=set)
    error: Optional[str] = None


@dataclass
class AgentEvent:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
