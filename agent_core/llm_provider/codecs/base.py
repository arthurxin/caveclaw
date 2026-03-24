from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from agent_core.assistant_messages.types import AssistantMessage, Message

from ..api_registry import StreamOptions


class ProviderMessageEncoder(Protocol):
    namespace: str

    def encode_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        ...


class ProviderChunkDecoder(Protocol):
    namespace: str

    def decode_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        ...

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        ...

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        ...


class ProviderMessageCodec(Protocol):
    namespace: str
    encoder: ProviderMessageEncoder
    decoder: ProviderChunkDecoder

    def encode_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        ...

    def decode_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        ...

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        ...

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        ...

    # Compatibility shims for the older method names.
    def to_provider_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        ...

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        ...


class BaseProviderMessageEncoder:
    namespace = "default"

    def encode_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        return [self._message_to_payload(message) for message in messages]

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state and isinstance(namespaced_state.get("replay_payload"), dict):
            return dict(namespaced_state["replay_payload"])
        return message.to_dict()


class BaseProviderChunkDecoder:
    namespace = "default"

    def decode_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        return dict(chunk)

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        return None

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        return None


class ComposedProviderMessageCodec:
    def __init__(
        self,
        *,
        namespace: str,
        encoder: Optional[ProviderMessageEncoder] = None,
        decoder: Optional[ProviderChunkDecoder] = None,
    ):
        self.namespace = namespace
        self.encoder = encoder or BaseProviderMessageEncoder()
        self.decoder = decoder or BaseProviderChunkDecoder()

    def encode_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        return self.encoder.encode_messages(messages, options)

    def decode_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        return self.decoder.decode_chunk(chunk, assistant_message)

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        self.decoder.finalize_provider_state(assistant_message)

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        self.decoder.finalize_assistant_message(assistant_message)

    # Compatibility wrappers
    def to_provider_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        return self.encode_messages(messages, options)

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        return self.decode_chunk(chunk, assistant_message)
