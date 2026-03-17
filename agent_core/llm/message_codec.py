from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from agent_core.types import AgentMessage, AssistantMessage, Message, ToolResultMessage

from .api_registry import StreamOptions


class ProviderMessageCodec(Protocol):
    namespace: str

    def to_provider_messages(
        self,
        messages: List[Message],
        options: StreamOptions,
    ) -> List[Dict[str, Any]]:
        ...


class DefaultMessageCodec:
    namespace = "default"

    def to_provider_messages(
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


class MiniMaxMessageCodec(DefaultMessageCodec):
    namespace = "minimax"


class GoogleMessageCodec(DefaultMessageCodec):
    namespace = "gemini"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)
        namespaced_state = message.get_provider_state(self.namespace)
        if not namespaced_state:
            return payload

        # Keep AgentMessage uniform while letting the provider reconstruct richer Gemini history.
        if "parts" in namespaced_state:
            payload["provider_state"] = {self.namespace: {"parts": namespaced_state["parts"]}}
        if "thought_signatures" in namespaced_state:
            payload.setdefault("provider_state", {}).setdefault(self.namespace, {})["thought_signatures"] = list(
                namespaced_state["thought_signatures"]
            )
        return payload


def codec_for_provider(provider: Any) -> ProviderMessageCodec:
    codec = getattr(provider, "message_codec", None)
    if codec is not None:
        return codec
    return DefaultMessageCodec()
