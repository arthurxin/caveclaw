from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agent_core.assistant_messages.types import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultMessage,
    content_blocks_to_text,
)

from .base import BaseProviderChunkDecoder, BaseProviderMessageEncoder, ComposedProviderMessageCodec


class DefaultMessageEncoder(BaseProviderMessageEncoder):
    namespace = "default"


class DefaultChunkDecoder(BaseProviderChunkDecoder):
    namespace = "default"


def _stringify_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, ensure_ascii=False)


def _render_content_block_as_text(block: ContentBlock) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ThinkingBlock):
        return block.thinking
    if isinstance(block, ToolResultBlock):
        return content_blocks_to_text(block.content_blocks)
    return content_blocks_to_text([block])


def _render_openai_content(
    message: Message,
    *,
    prefer_raw_content: bool = False,
) -> str | List[Dict[str, Any]]:
    if prefer_raw_content and message.raw_content is not None:
        return message.raw_content

    parts: List[Dict[str, Any]] = []
    saw_non_text_block = False
    for block in message.content_blocks:
        if isinstance(block, ImageBlock):
            if block.image_url:
                saw_non_text_block = True
                image_payload: Dict[str, Any] = {"url": block.image_url}
                if block.mime_type:
                    image_payload["mime_type"] = block.mime_type
                parts.append({"type": "image_url", "image_url": image_payload})
            continue

        text = _render_content_block_as_text(block)
        if not text:
            continue
        if isinstance(block, TextBlock) and not saw_non_text_block and not parts:
            parts.append({"type": "text", "text": text})
            continue
        if not isinstance(block, TextBlock):
            saw_non_text_block = True
        if parts and parts[-1].get("type") == "text":
            parts[-1]["text"] += text
        else:
            parts.append({"type": "text", "text": text})

    if not parts:
        return message.raw_content if message.raw_content is not None else message.content
    if len(parts) == 1 and parts[0].get("type") == "text" and not saw_non_text_block:
        return parts[0]["text"]
    return parts


def _render_minimax_replay_content(message: Message) -> str | List[Dict[str, Any]]:
    if message.raw_content is not None:
        return message.raw_content

    thinking_parts: List[str] = []
    visible_parts: List[str] = []
    for block in message.content_blocks:
        if isinstance(block, ThinkingBlock):
            if block.thinking:
                thinking_parts.append(block.thinking)
            continue
        rendered = _render_content_block_as_text(block)
        if rendered:
            visible_parts.append(rendered)

    payload_parts: List[str] = []
    if thinking_parts:
        payload_parts.append(f"<think>{''.join(thinking_parts)}</think>")
    if visible_parts:
        payload_parts.append("\n".join(visible_parts))
    if payload_parts:
        return "\n".join(payload_parts)
    return _render_openai_content(message, prefer_raw_content=False)


def _parse_data_url(image_url: Optional[str], mime_type: Optional[str] = None) -> Optional[Tuple[str, str]]:
    if not image_url or not image_url.startswith("data:"):
        return None
    header, sep, data = image_url.partition(",")
    if not sep:
        return None
    metadata = header[5:]
    parts = [part for part in metadata.split(";") if part]
    media_type = mime_type or (parts[0] if parts else None) or "image/png"
    if "base64" not in parts:
        return None
    return media_type, data


def _render_image_fallback_text(block: ImageBlock) -> str:
    if block.alt_text:
        return f"[Image: {block.alt_text}]"
    if block.image_url:
        return f"[Image: {block.image_url}]"
    return "[Image]"


def _infer_google_parts_from_blocks(blocks: List[ContentBlock]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    text_segments: List[str] = []

    def flush_text() -> None:
        if text_segments:
            parts.append({"text": "".join(text_segments)})
            text_segments.clear()

    for block in blocks:
        if isinstance(block, ImageBlock):
            parsed = _parse_data_url(block.image_url, block.mime_type)
            if parsed:
                flush_text()
                media_type, data = parsed
                parts.append({"inlineData": {"mimeType": media_type, "data": data}})
            else:
                text_segments.append(_render_image_fallback_text(block))
            continue

        rendered = _render_content_block_as_text(block)
        if rendered:
            text_segments.append(rendered)

    flush_text()
    return parts


def _render_anthropic_content(message: Message) -> str | List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    text_segments: List[str] = []

    def flush_text() -> None:
        if text_segments:
            parts.append({"type": "text", "text": "".join(text_segments)})
            text_segments.clear()

    for block in message.content_blocks:
        if isinstance(block, ImageBlock):
            parsed = _parse_data_url(block.image_url, block.mime_type)
            if parsed:
                flush_text()
                media_type, data = parsed
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                text_segments.append(_render_image_fallback_text(block))
            continue

        rendered = _render_content_block_as_text(block)
        if rendered:
            text_segments.append(rendered)

    flush_text()
    if not parts:
        return message.content
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _assistant_message_to_anthropic_content_blocks(message: AssistantMessage) -> List[Dict[str, Any]]:
    content_blocks: List[Dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()

    def append_text(text: str) -> None:
        if not text:
            return
        if content_blocks and content_blocks[-1].get("type") == "text":
            content_blocks[-1]["text"] += text
        else:
            content_blocks.append({"type": "text", "text": text})

    for block in message.content_blocks:
        if isinstance(block, TextBlock):
            append_text(block.text)
            continue
        if isinstance(block, ThinkingBlock):
            if block.thinking:
                payload: Dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
                if block.signature:
                    payload["signature"] = block.signature
                content_blocks.append(payload)
            continue
        if isinstance(block, ImageBlock):
            rendered = _render_anthropic_content(Message(role="assistant", content_blocks=[block]))
            if isinstance(rendered, list):
                content_blocks.extend(rendered)
            elif rendered:
                append_text(rendered)
            continue
        if isinstance(block, ToolCallBlock):
            seen_tool_call_ids.add(block.id)
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.arguments),
                }
            )
            continue

        rendered = _render_content_block_as_text(block)
        if rendered:
            append_text(rendered)

    for tool_call in message.tool_calls or []:
        if tool_call.id in seen_tool_call_ids:
            continue
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": dict(tool_call.arguments),
            }
        )

    return content_blocks


class OpenAICompatibleMessageEncoder(BaseProviderMessageEncoder):
    namespace = "openai"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state and isinstance(namespaced_state.get("replay_payload"), dict):
            return dict(namespaced_state["replay_payload"])

        if isinstance(message, ToolResultMessage):
            payload = {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.raw_content if message.raw_content is not None else message.content,
            }
            if message.name:
                payload["name"] = message.name
            return payload

        payload: Dict[str, Any] = {"role": message.role}
        payload["content"] = _render_openai_content(
            message,
            prefer_raw_content=isinstance(message, AssistantMessage),
        )

        if isinstance(message, AssistantMessage) and message.tool_calls:
            payload["tool_calls"] = []
            for tool_call in message.tool_calls:
                payload["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": _stringify_tool_arguments(tool_call.arguments),
                        },
                    }
                )
        return payload


class MiniMaxChunkDecoder(DefaultChunkDecoder):
    namespace = "minimax"

    def decode_chunk(
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


class MiniMaxMessageEncoder(OpenAICompatibleMessageEncoder):
    namespace = "minimax"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state and isinstance(namespaced_state.get("replay_payload"), dict):
            return dict(namespaced_state["replay_payload"])

        payload = super()._message_to_payload(message)
        if payload.get("role") == "assistant":
            payload["content"] = _render_minimax_replay_content(message)
        return payload


class ArkMessageEncoder(OpenAICompatibleMessageEncoder):
    namespace = "ark"


class AzureMessageEncoder(DefaultMessageEncoder):
    namespace = "azure"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)
        namespaced_state = message.get_provider_state(self.namespace)
        if namespaced_state:
            payload["provider_state"] = {self.namespace: dict(namespaced_state)}
        return payload


class GoogleMessageEncoder(DefaultMessageEncoder):
    namespace = "gemini"

    def _message_to_payload(self, message: Message) -> Dict[str, Any]:
        payload = super()._message_to_payload(message)
        namespaced_state = message.get_provider_state(self.namespace) or {}
        provider_parts = [self._part_to_provider_payload(part) for part in namespaced_state.get("parts", [])]

        if not provider_parts:
            provider_parts = self._infer_parts_from_message(message)

        if not provider_parts and not namespaced_state:
            return payload

        if provider_parts:
            payload["provider_state"] = {self.namespace: {"parts": provider_parts}}
        if "thought_signatures" in namespaced_state:
            payload.setdefault("provider_state", {}).setdefault(self.namespace, {})["thought_signatures"] = list(
                namespaced_state["thought_signatures"]
            )
        return payload

    def _infer_parts_from_message(self, message: Message) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        if isinstance(message, AssistantMessage):
            parts.extend(_infer_google_parts_from_blocks(message.content_blocks))
            if not parts and message.content:
                parts.append({"text": message.content})
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

        parts.extend(_infer_google_parts_from_blocks(message.content_blocks))
        if parts:
            return parts
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


class GoogleChunkDecoder(DefaultChunkDecoder):
    namespace = "gemini"

    def decode_chunk(
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

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        namespaced_state = assistant_message.get_provider_state(self.namespace)
        if not namespaced_state:
            return
        updated_state = dict(namespaced_state)
        thought_signatures = updated_state.get("thought_signatures")
        if isinstance(thought_signatures, list):
            updated_state["thought_signatures"] = list(dict.fromkeys(thought_signatures))

        provider_parts = updated_state.get("parts")
        if isinstance(provider_parts, list):
            thought_parts = [
                self._part_to_provider_payload(part)
                for part in provider_parts
                if isinstance(part, dict) and part.get("thoughtSignature")
            ]
            if thought_parts:
                updated_state["thought_parts"] = thought_parts
        assistant_message.set_provider_state(self.namespace, updated_state)

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        namespaced_state = assistant_message.get_provider_state(self.namespace)
        if not namespaced_state:
            return

        thought_signatures = namespaced_state.get("thought_signatures")
        if isinstance(thought_signatures, list) and thought_signatures:
            thinking_blocks = [block for block in assistant_message.content_blocks if isinstance(block, ThinkingBlock)]
            if thinking_blocks and not thinking_blocks[-1].signature:
                thinking_blocks[-1].signature = str(thought_signatures[-1])

    def _part_to_provider_payload(self, part: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(part)
        if "function_call" in normalized and "functionCall" not in normalized:
            normalized["functionCall"] = normalized.pop("function_call")
        if "function_response" in normalized and "functionResponse" not in normalized:
            normalized["functionResponse"] = normalized.pop("function_response")
        if "thought_signature" in normalized and "thoughtSignature" not in normalized:
            normalized["thoughtSignature"] = normalized.pop("thought_signature")
        return normalized


class AnthropicChunkDecoder(DefaultChunkDecoder):
    namespace = "anthropic"

    def finalize_provider_state(self, assistant_message: AssistantMessage) -> None:
        namespaced_state = assistant_message.get_provider_state(self.namespace)
        if not namespaced_state:
            return

        updated_state = dict(namespaced_state)
        thought_signatures = updated_state.get("thought_signatures")
        if isinstance(thought_signatures, list):
            updated_state["thought_signatures"] = list(dict.fromkeys(thought_signatures))
        assistant_message.set_provider_state(self.namespace, updated_state)

    def finalize_assistant_message(self, assistant_message: AssistantMessage) -> None:
        namespaced_state = assistant_message.get_provider_state(self.namespace) or {}
        updated_state = dict(namespaced_state)

        thought_signatures = updated_state.get("thought_signatures")
        if isinstance(thought_signatures, list) and thought_signatures:
            thinking_blocks = [block for block in assistant_message.content_blocks if isinstance(block, ThinkingBlock)]
            if thinking_blocks and not thinking_blocks[-1].signature:
                thinking_blocks[-1].signature = str(thought_signatures[-1])

        if "replay_payload" not in updated_state:
            replay_content = _assistant_message_to_anthropic_content_blocks(assistant_message)
            updated_state["replay_payload"] = {
                "role": "assistant",
                "content": replay_content or assistant_message.content,
            }

        if updated_state:
            assistant_message.set_provider_state(self.namespace, updated_state)


class AnthropicMessageEncoder(DefaultMessageEncoder):
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
            content_blocks = _assistant_message_to_anthropic_content_blocks(message)
            return {
                "role": "assistant",
                "content": content_blocks,
            }

        return {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": _render_anthropic_content(message),
        }


class DefaultMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="default",
            encoder=DefaultMessageEncoder(),
            decoder=DefaultChunkDecoder(),
        )


class OpenAICompatibleMessageCodec(ComposedProviderMessageCodec):
    def __init__(self, namespace: str = "openai"):
        encoder = OpenAICompatibleMessageEncoder()
        encoder.namespace = namespace
        decoder = DefaultChunkDecoder()
        decoder.namespace = namespace
        super().__init__(
            namespace=namespace,
            encoder=encoder,
            decoder=decoder,
        )


class MiniMaxMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="minimax",
            encoder=MiniMaxMessageEncoder(),
            decoder=MiniMaxChunkDecoder(),
        )


class ArkMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="ark",
            encoder=ArkMessageEncoder(),
            decoder=DefaultChunkDecoder(),
        )


class AzureMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="azure",
            encoder=AzureMessageEncoder(),
            decoder=DefaultChunkDecoder(),
        )


class GoogleMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="gemini",
            encoder=GoogleMessageEncoder(),
            decoder=GoogleChunkDecoder(),
        )


class AnthropicMessageCodec(ComposedProviderMessageCodec):
    def __init__(self):
        super().__init__(
            namespace="anthropic",
            encoder=AnthropicMessageEncoder(),
            decoder=AnthropicChunkDecoder(),
        )
