from __future__ import annotations

from typing import Any

from .codecs import (
    AnthropicMessageCodec,
    ArkMessageCodec,
    AzureMessageCodec,
    DefaultMessageCodec,
    GoogleMessageCodec,
    MiniMaxMessageCodec,
    OpenAICompatibleMessageCodec,
    ProviderMessageCodec,
)


def codec_for_provider(provider: Any) -> ProviderMessageCodec:
    codec = getattr(provider, "message_codec", None)
    if codec is not None:
        return codec
    return DefaultMessageCodec()


__all__ = [
    "AnthropicMessageCodec",
    "ArkMessageCodec",
    "AzureMessageCodec",
    "DefaultMessageCodec",
    "GoogleMessageCodec",
    "MiniMaxMessageCodec",
    "OpenAICompatibleMessageCodec",
    "ProviderMessageCodec",
    "codec_for_provider",
]
