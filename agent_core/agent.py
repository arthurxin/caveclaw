from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, List, Optional

from .agent_loop import run_loop
from .types import AgentEvent, AgentLoopConfig, AgentMessage, AgentState, AgentTool, BasicCancellationSignal, RuntimeState


class Agent:
    """High-level facade around the core loop and message queues."""

    def __init__(self, config: AgentLoopConfig, tools: Optional[List[AgentTool]] = None):
        self.state = AgentState(tools=tools or [])
        self.config = config

        self._steering_queue: List[AgentMessage] = []
        self._followup_queue: List[AgentMessage] = []
        self._steering_mode: str = "one-at-a-time"
        self._followup_mode: str = "one-at-a-time"
        self._abort_requested = False
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._listeners: List[Callable[[AgentEvent], None]] = []
        self._active_abort_signal: Optional[BasicCancellationSignal] = None

        self._original_get_steering = getattr(config, "get_steering_messages", None)
        setattr(config, "get_steering_messages", self._get_steering_messages)

        self._original_get_followup = getattr(config, "get_followup_messages", None)
        setattr(config, "get_followup_messages", self._get_followup_messages)

    async def _get_steering_messages(self) -> List[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            msgs = self._steering_queue[:1]
            self._steering_queue = self._steering_queue[1:]
        else:
            msgs = self._steering_queue.copy()
            self._steering_queue.clear()

        if self._original_get_steering:
            extra = await self._original_get_steering()
            msgs.extend(extra)
        return msgs

    async def _get_followup_messages(self) -> List[AgentMessage]:
        if self._followup_mode == "one-at-a-time":
            msgs = self._followup_queue[:1]
            self._followup_queue = self._followup_queue[1:]
        else:
            msgs = self._followup_queue.copy()
            self._followup_queue.clear()

        if self._original_get_followup:
            extra = await self._original_get_followup()
            msgs.extend(extra)
        return msgs

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def set_system_prompt(self, prompt: str):
        self.state.system_prompt = prompt
        setattr(self.config, "system_prompt", prompt)

    def set_model(self, model) -> None:
        self.state.model = model
        setattr(self.config, "model", model)

    def set_thinking_level(self, level: str) -> None:
        self.state.thinking_level = level
        setattr(self.config, "thinking_level", level)

    def set_tools(self, tools: List[AgentTool]) -> None:
        self.state.tools = list(tools)

    def set_runtime(self, runtime: RuntimeState) -> None:
        self.state.runtime = runtime

    def replace_messages(self, messages: List[AgentMessage]):
        self.state.messages = list(messages)

    def append_message(self, message: AgentMessage):
        self.state.messages.append(message)

    def clear_messages(self) -> None:
        self.state.messages = []

    def steer(self, message: AgentMessage):
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage):
        self._followup_queue.append(message)

    def clear_queues(self):
        self._steering_queue.clear()
        self._followup_queue.clear()

    def clear_all_queues(self):
        self.clear_queues()

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._followup_queue)

    def set_steering_mode(self, mode: str) -> None:
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        return self._steering_mode

    def set_followup_mode(self, mode: str) -> None:
        self._followup_mode = mode

    def get_followup_mode(self) -> str:
        return self._followup_mode

    def abort(self) -> None:
        self._abort_requested = True
        if self._active_abort_signal is not None:
            self._active_abort_signal.cancel("Agent abort requested")

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    async def continue_run(
        self,
        stream_fn: Optional[Callable] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        if not self.state.messages:
            raise ValueError("Cannot continue without any messages.")
        if getattr(self.state.messages[-1], "role", None) == "assistant":
            raise ValueError("Cannot continue from a final assistant message.")
        async for event in self._drive_loop(stream_fn=stream_fn):
            yield event

    async def continue_(self, stream_fn: Optional[Callable] = None) -> AsyncGenerator[AgentEvent, None]:
        async for event in self.continue_run(stream_fn=stream_fn):
            yield event

    async def prompt(
        self,
        prompt: AgentMessage,
        stream_fn: Optional[Callable] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.state.messages.append(prompt)
        async for event in self._drive_loop(stream_fn=stream_fn):
            yield event

    async def _drive_loop(
        self,
        stream_fn: Optional[Callable] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        self._abort_requested = False
        self.state.is_streaming = True
        self.state.stream_message = None
        self._idle_event.clear()
        self._active_abort_signal = BasicCancellationSignal()
        setattr(self.config, "abort_signal", self._active_abort_signal)

        try:
            async for event in run_loop(
                self.state.messages,
                self.state.tools,
                self.config,
                stream_fn,
                runtime=self.state.runtime,
            ):
                event_message = event.data.get("message") if isinstance(event.data, dict) else None
                if event.type == "message_update" and event_message is not None:
                    self.state.stream_message = event_message
                elif event.type == "message_end" and event_message is not None:
                    self.state.stream_message = None
                elif event.type == "agent_end":
                    self.state.messages = list(event.data.get("messages", self.state.messages))

                self._emit(event)
                yield event

                if self._abort_requested and event.type in {"tool_execution_success", "tool_execution_error", "turn_end"}:
                    break
        finally:
            self.state.is_streaming = False
            self.state.stream_message = None
            self._active_abort_signal = None
            self._idle_event.set()
