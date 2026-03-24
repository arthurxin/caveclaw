from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, List, Optional, Sequence

from agent_core import Agent, AgentEvent, AgentMessage, AgentTool, Message
from agent_core.llm_provider import Model


class MainChatAgentConfig:
    """Minimal config substrate for a top-level chat agent prototype."""

    max_rounds = 20
    max_consecutive_tool_failures = 5

    def __init__(self, *, model: Model, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.thinking_level = "off"
        self.temperature = None
        self.max_tokens = None
        self.session_id = None
        self.transport = None
        self.cache_retention = None
        self.headers = None
        self.max_retry_delay_ms = None
        self.metadata = None
        self.thinking_budgets = None
        self.on_payload = None

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        return [message for message in messages if isinstance(message, Message)]

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        return messages

    async def get_steering_messages(self) -> List[AgentMessage]:
        return []

    async def get_followup_messages(self) -> List[AgentMessage]:
        return []


class MainChatAgent(ABC):
    """
    Abstract blueprint for the top-level user-facing chat agent.

    This layer stays above `agent_core` and is intentionally thin:
    it defines how a main input agent is assembled, without baking in
    concrete product workflows or domain-specific business logic.
    """

    def __init__(
        self,
        *,
        model: Optional[Model] = None,
        tools: Optional[Sequence[AgentTool]] = None,
        messages: Optional[Sequence[AgentMessage]] = None,
    ):
        resolved_model = model or self.build_default_model()
        resolved_tools = list(tools) if tools is not None else list(self.build_tools())
        config = self.build_config(resolved_model)
        self.config = config
        self.agent = Agent(
            config,
            tools=resolved_tools,
            messages=list(messages or []),
        )

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Return the system prompt used by the main chat agent."""

    def build_tools(self) -> Sequence[AgentTool]:
        """Override to attach runtime-aware tools for this main chat agent."""
        return []

    def build_config(self, model: Model) -> MainChatAgentConfig:
        config = MainChatAgentConfig(
            model=model,
            system_prompt=self.build_system_prompt(),
        )
        self.configure_config(config)
        return config

    def configure_config(self, config: MainChatAgentConfig) -> None:
        """Override to enable python lanes, inspectors, or other policy hooks."""
        del config

    def build_default_model(self) -> Model:
        """
        Return a placeholder model descriptor for prototype wiring.

        Real applications can inject a fully resolved model from `ModelRegistry`
        or `ModelResolver` when instantiating a concrete subclass.
        """

        return Model(
            id="prototype-mainchat",
            provider="prototype",
            api="prototype-stream",
        )

    def build_user_message(self, user_input: str) -> Message:
        return Message(role="user", content=user_input)

    async def handle_user_input(
        self,
        user_input: str,
        *,
        stream_fn: Optional[Callable[..., AsyncGenerator[Any, None]]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        async for event in self.agent.prompt(
            self.build_user_message(user_input),
            stream_fn=stream_fn,
        ):
            yield event

    async def continue_dialogue(
        self,
        *,
        stream_fn: Optional[Callable[..., AsyncGenerator[Any, None]]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        async for event in self.agent.continue_run(stream_fn=stream_fn):
            yield event


class _ExampleMainChatAgent(MainChatAgent):
    def build_system_prompt(self) -> str:
        return (
            "You are the prototype main chat agent. "
            "Receive the user's top-level request, keep the conversation coherent, "
            "and respond clearly."
        )


async def _demo_stream(messages: List[Message]) -> AsyncGenerator[dict[str, Any], None]:
    last_user_message = next(
        (message for message in reversed(messages) if getattr(message, "role", None) == "user"),
        None,
    )
    user_content = last_user_message.content if last_user_message is not None else ""
    yield {"content": f"Prototype reply: {user_content}"}


async def _run_demo() -> None:
    agent = _ExampleMainChatAgent()
    async for event in agent.handle_user_input("Hello from mainchat_agent.", stream_fn=_demo_stream):
        if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
            print(event.message.content)


if __name__ == "__main__":
    asyncio.run(_run_demo())
