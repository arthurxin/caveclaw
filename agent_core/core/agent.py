from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterable, List, Optional, Sequence, Union

from .agent_loop import run_loop
from .handoff import HandoffOptions, handoff_messages
from .python_program_execution import IPythonProgramRuntime
from .session_context import AgentHostContext, AgentSessionContext, ApiKeyResolver
from ..assistant_messages.types import (
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    BasicCancellationSignal,
    Message,
    RuntimeState,
)

MessageInput = Union[AgentMessage, str]
MessageInputBatch = Union[MessageInput, Sequence[MessageInput]]


class Agent:
    """High-level facade around the core loop and message queues."""

    def __init__(
        self,
        config: AgentLoopConfig,
        tools: Optional[List[AgentTool]] = None,
        *,
        messages: Optional[List[AgentMessage]] = None,
        runtime: Optional[RuntimeState] = None,
        python_runtime: Optional[Any] = None,
        model_registry: Optional[Any] = None,
        api_key: Optional[str] = None,
        get_api_key: Optional[ApiKeyResolver] = None,
    ):
        resolved_model_registry = model_registry if model_registry is not None else getattr(config, "model_registry", None)
        host_context = AgentHostContext(
            runtime=runtime or RuntimeState(),
            python_runtime=python_runtime or IPythonProgramRuntime(),
            model_registry=resolved_model_registry,
            api_key=api_key,
            get_api_key=get_api_key,
        )
        self.state = AgentState(
            system_prompt=getattr(config, "system_prompt", "") or "",
            model=getattr(config, "model", "gpt-4"),
            thinking_level=getattr(config, "thinking_level", "off") or "off",
            tools=list(tools or []),
            messages=list(messages or []),
            runtime=host_context.runtime,
            python_runtime=host_context.python_runtime,
            host_context=host_context,
            session_id=getattr(config, "session_id", None),
            transport=getattr(config, "transport", None),
            cache_retention=getattr(config, "cache_retention", None),
            temperature=getattr(config, "temperature", None),
            max_tokens=getattr(config, "max_tokens", None),
            headers=getattr(config, "headers", None),
            max_retry_delay_ms=getattr(config, "max_retry_delay_ms", None),
            metadata=getattr(config, "metadata", None),
            thinking_budgets=getattr(config, "thinking_budgets", None),
            on_payload=getattr(config, "on_payload", None),
        )
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
        self._api_key_resolver: Optional[ApiKeyResolver] = get_api_key

        self._original_get_steering = getattr(config, "get_steering_messages", None)
        setattr(config, "get_steering_messages", self._get_steering_messages)

        self._original_get_followup = getattr(config, "get_followup_messages", None)
        setattr(config, "get_followup_messages", self._get_followup_messages)

        self._original_get_api_key = getattr(config, "get_api_key", None)
        setattr(config, "get_api_key", self._get_api_key)

        if resolved_model_registry is not None:
            setattr(config, "model_registry", resolved_model_registry)
        if api_key is not None:
            self.set_api_key(api_key)

    async def _get_api_key(self, provider: str) -> Optional[str]:
        resolver = self.state.host_context.get_api_key if self.state.host_context is not None else self._api_key_resolver
        if resolver is not None:
            resolved = resolver(provider)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved:
                return str(resolved)

        if self._original_get_api_key is not None:
            resolved = self._original_get_api_key(provider)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved:
                return str(resolved)

        registry = self.state.host_context.model_registry if self.state.host_context is not None else getattr(self.config, "model_registry", None)
        if registry is not None and hasattr(registry, "get_api_key"):
            resolved = registry.get_api_key(provider)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if resolved:
                return str(resolved)

        configured_api_key = self.state.host_context.api_key if self.state.host_context is not None else getattr(self.config, "api_key", None)
        if configured_api_key:
            return str(configured_api_key)
        return None

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

    def handoff_to_model(
        self,
        model,
        *,
        preserve_provider_state: bool = False,
    ) -> None:
        result = handoff_messages(
            self.state.messages,
            model,
            options=HandoffOptions(
                provider_state_policy="preserve" if preserve_provider_state else "same-model-only",
            ),
        )
        self.replace_messages(result.messages)
        self.set_model(model)

    def set_thinking_level(self, level: str) -> None:
        self.state.thinking_level = level
        setattr(self.config, "thinking_level", level)

    def set_temperature(self, temperature: Optional[float]) -> None:
        self.state.temperature = temperature
        setattr(self.config, "temperature", temperature)

    def set_max_tokens(self, max_tokens: Optional[int]) -> None:
        self.state.max_tokens = max_tokens
        setattr(self.config, "max_tokens", max_tokens)

    def set_session_id(self, session_id: Optional[str]) -> None:
        self.state.session_id = session_id
        setattr(self.config, "session_id", session_id)

    def set_transport(self, transport: Optional[str]) -> None:
        self.state.transport = transport
        setattr(self.config, "transport", transport)

    def set_cache_retention(self, cache_retention: Optional[str]) -> None:
        self.state.cache_retention = cache_retention
        setattr(self.config, "cache_retention", cache_retention)

    def set_headers(self, headers: Optional[dict[str, str]]) -> None:
        self.state.headers = dict(headers) if headers else None
        setattr(self.config, "headers", dict(headers) if headers else None)

    def set_max_retry_delay_ms(self, max_retry_delay_ms: Optional[int]) -> None:
        self.state.max_retry_delay_ms = max_retry_delay_ms
        setattr(self.config, "max_retry_delay_ms", max_retry_delay_ms)

    def set_request_metadata(self, metadata: Optional[dict[str, Any]]) -> None:
        self.state.metadata = dict(metadata) if metadata else None
        setattr(self.config, "metadata", dict(metadata) if metadata else None)

    def set_thinking_budgets(self, thinking_budgets: Optional[dict[str, Any]]) -> None:
        self.state.thinking_budgets = dict(thinking_budgets) if thinking_budgets else None
        setattr(self.config, "thinking_budgets", dict(thinking_budgets) if thinking_budgets else None)

    def set_on_payload(self, on_payload: Optional[Callable[..., Any]]) -> None:
        self.state.on_payload = on_payload
        setattr(self.config, "on_payload", on_payload)

    def set_tools(self, tools: List[AgentTool]) -> None:
        self.state.tools = list(tools)

    def set_runtime(self, runtime: RuntimeState) -> None:
        self.state.runtime = runtime
        if self.state.host_context is not None:
            self.state.host_context.runtime = runtime

    def set_python_runtime(self, python_runtime: Any) -> None:
        self.state.python_runtime = python_runtime
        if self.state.host_context is not None:
            self.state.host_context.python_runtime = python_runtime

    def set_model_registry(self, model_registry: Any) -> None:
        if self.state.host_context is not None:
            self.state.host_context.model_registry = model_registry
        setattr(self.config, "model_registry", model_registry)

    def set_api_key(self, api_key: str) -> None:
        if self.state.host_context is not None:
            self.state.host_context.api_key = api_key
        setattr(self.config, "api_key", api_key)

    def set_api_key_resolver(
        self,
        resolver: Optional[ApiKeyResolver],
    ) -> None:
        self._api_key_resolver = resolver
        if self.state.host_context is not None:
            self.state.host_context.get_api_key = resolver

    def get_host_context(self) -> AgentHostContext:
        if self.state.host_context is None:
            self.state.host_context = AgentHostContext(
                runtime=self.state.runtime,
                python_runtime=self.state.python_runtime or IPythonProgramRuntime(),
                model_registry=getattr(self.config, "model_registry", None),
                api_key=getattr(self.config, "api_key", None),
                get_api_key=self._api_key_resolver,
            )
        return self.state.host_context

    def set_host_context(self, host_context: AgentHostContext) -> None:
        self.state.host_context = host_context
        self.state.runtime = host_context.runtime
        self.state.python_runtime = host_context.python_runtime
        if host_context.model_registry is not None:
            setattr(self.config, "model_registry", host_context.model_registry)
        setattr(self.config, "api_key", host_context.api_key)
        self._api_key_resolver = host_context.get_api_key

    def export_session_context(self) -> AgentSessionContext:
        return AgentSessionContext(
            messages=list(self.state.messages),
            tools=list(self.state.tools),
            host=self.get_host_context(),
        )

    def replace_session_context(self, session_context: AgentSessionContext) -> None:
        self.state.messages = list(session_context.messages)
        self.state.tools = list(session_context.tools)
        self.set_host_context(session_context.host)

    def replace_messages(self, messages: List[AgentMessage]):
        self.state.messages = list(messages)

    def append_message(self, message: AgentMessage):
        self.state.messages.append(message)

    def clear_messages(self) -> None:
        self.state.messages = []

    @staticmethod
    def _coerce_message(message: MessageInput) -> AgentMessage:
        if isinstance(message, str):
            return Message(role="user", content=message)
        return message

    @classmethod
    def _coerce_messages(cls, messages: MessageInputBatch) -> List[AgentMessage]:
        if isinstance(messages, str):
            return [cls._coerce_message(messages)]
        if isinstance(messages, Sequence):
            return [cls._coerce_message(message) for message in messages]
        return [cls._coerce_message(messages)]

    def steer(self, message: MessageInputBatch):
        self._steering_queue.extend(self._coerce_messages(message))

    def follow_up(self, message: MessageInputBatch):
        self._followup_queue.extend(self._coerce_messages(message))

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
        prompt: MessageInputBatch,
        stream_fn: Optional[Callable] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.state.messages.extend(self._coerce_messages(prompt))
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
                python_runtime=self.state.python_runtime,
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
