from __future__ import annotations

import json
from typing import Any, Dict, List, Protocol

from agent_core.assistant_messages.types import AgentMessage, AssistantMessage, Message, ToolResultMessage

from .api_registry import StreamOptions


class ProviderMessageCodec(Protocol):
    namespace: str

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

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
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

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        return dict(chunk)

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        return None


class MiniMaxMessageCodec(DefaultMessageCodec):
    namespace = "minimax"

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        normalized = dict(chunk)
        if "raw_content" in normalized:
            normalized.setdefault("provider_state", {})
            normalized["provider_state"][self.namespace] = {
                "replay_payload": {
                    "role": "assistant",
                    "content": normalized["raw_content"],
                }
            }
        return normalized


class ArkMessageCodec(DefaultMessageCodec):
    namespace = "ark"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)

        if isinstance(message, AssistantMessage) and message.tool_calls:
            payload["tool_calls"] = []
            for tool_call in message.tool_calls:
                payload["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                            if isinstance(tool_call.arguments, str)
                            else json.dumps(tool_call.arguments, ensure_ascii=False),
                        },
                    }
                )
        return payload


class AzureMessageCodec(DefaultMessageCodec):
    namespace = "azure"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state:
            payload["provider_state"] = {self.namespace: dict(namespaced_state)}
        return payload


class GoogleMessageCodec(DefaultMessageCodec):
    namespace = "gemini"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)
        namespaced_state = message.get_provider_state(self.namespace) or {}
        provider_parts = [self._part_to_provider_payload(part) for part in namespaced_state.get("parts", [])]

        if not provider_parts:
            provider_parts = self._infer_parts_from_message(message)

        if not provider_parts and not namespaced_state:
            return payload

        # Keep AgentMessage uniform while letting the provider reconstruct richer Gemini history.
        if provider_parts:
            payload["provider_state"] = {self.namespace: {"parts": provider_parts}}
        if "thought_signatures" in namespaced_state:
            payload.setdefault("provider_state", {}).setdefault(self.namespace, {})["thought_signatures"] = list(
                namespaced_state["thought_signatures"]
            )
        return payload

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        normalized = dict(chunk)
        provider_state = normalized.get("provider_state")
        if isinstance(provider_state, dict) and isinstance(provider_state.get(self.namespace), dict):
            namespaced = dict(provider_state[self.namespace])
            if "thought_signatures" in namespaced and isinstance(namespaced["thought_signatures"], list):
                namespaced["thought_signatures"] = list(dict.fromkeys(namespaced["thought_signatures"]))
            normalized["provider_state"] = {self.namespace: namespaced}
        return normalized

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        namespaced_state = assistant_message.get_provider_state(self.namespace)
        if not namespaced_state:
            return
        thought_signatures = namespaced_state.get("thought_signatures")
        if isinstance(thought_signatures, list):
            assistant_message.set_provider_state(
                self.namespace,
                {
                    **namespaced_state,
                    "thought_signatures": list(dict.fromkeys(thought_signatures)),
                },
            )

    def _infer_parts_from_message(self, message: Message) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        if isinstance(message, AssistantMessage):
            content = message.content
            if content:
                parts.append({"text": content})
            for tool_call in message.tool_calls or []:
                parts.append(
                    {
                        "functionCall": {
                            "name": tool_call.name,
                            "args": dict(tool_call.arguments),
                        }
                    }
                )
            return parts

        if isinstance(message, ToolResultMessage):
            return [
                {
                    "functionResponse": {
                        "name": message.name,
                        "response": {
                            "result": message.content,
                        },
                    }
                }
            ]

        if message.content:
            return [{"text": message.content}]
        return parts

    def _part_to_provider_payload(self, part: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(part)
        if "function_call" in normalized and "functionCall" not in normalized:
            normalized["functionCall"] = normalized.pop("function_call")
        if "function_response" in normalized and "functionResponse" not in normalized:
            normalized["functionResponse"] = normalized.pop("function_response")
        if "thought_signature" in normalized and "thoughtSignature" not in normalized:
            normalized["thoughtSignature"] = normalized.pop("thought_signature")

        function_call = normalized.get("functionCall")
        if isinstance(function_call, dict) and "args" not in function_call and isinstance(function_call.get("arguments"), dict):
            converted = dict(function_call)
            converted["args"] = converted.pop("arguments")
            normalized["functionCall"] = converted

        return normalized


class AnthropicMessageCodec(DefaultMessageCodec):
    namespace = "anthropic"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state and isinstance(namespaced_state.get("replay_payload"), dict):
            return dict(namespaced_state["replay_payload"])

        if isinstance(message, ToolResultMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }

        if isinstance(message, AssistantMessage) and message.tool_calls:
            content_blocks: List[Dict[str, Any]] = []
            if message.content:
                content_blocks.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": dict(tool_call.arguments),
                    }
                )
            return {
                "role": "assistant",
                "content": content_blocks,
            }

        return {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": message.content,
        }

    def from_provider_chunk(
        self,
        chunk: Dict[str, Any],
        assistant_message: AssistantMessage,
    ) -> Dict[str, Any]:
        return dict(chunk)


def codec_for_provider(provider: Any) -> ProviderMessageCodec:
    codec = getattr(provider, "message_codec", None)
    if codec is not None:
        return codec
    return DefaultMessageCodec()
