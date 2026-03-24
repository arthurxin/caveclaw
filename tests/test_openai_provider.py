import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import RateLimitError

from agent_core.assistant_messages.tools import AgentTool
from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.provider_types import Model, ModelCompat
from agent_core.llm_provider.providers.openai_provider import OpenAiProvider


class OpenAiProviderTests(unittest.TestCase):
    def test_stream_retries_when_delay_is_within_cap(self):
        model = Model(id="gpt-5.4", provider="openai", api="openai-chat", maxTokens=1024)
        provider = OpenAiProvider()
        options = StreamOptions(max_retry_delay_ms=1500)

        class FakeStream:
            def __init__(self, chunks):
                self._chunks = chunks

            def __aiter__(self):
                self._iterator = iter(self._chunks)
                return self

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        async def collect():
            with patch("openai.AsyncOpenAI") as mock_client_cls:
                request = httpx.Request("POST", "https://example.com/v1/chat/completions")
                response = httpx.Response(429, headers={"Retry-After": "0"}, request=request)
                rate_limit_error = RateLimitError(
                    "Please retry in 0s",
                    response=response,
                    body={"error": {"message": "Please retry in 0s"}},
                )

                calls = {"count": 0}

                async def fake_create(**kwargs):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise rate_limit_error
                    return FakeStream(
                        [
                            SimpleNamespace(
                                usage=None,
                                choices=[SimpleNamespace(delta=SimpleNamespace(content="hello", tool_calls=None))],
                            )
                        ]
                    )

                mock_client = mock_client_cls.return_value
                mock_client.chat.completions.create.side_effect = fake_create

                chunks = []
                with patch("agent_core.llm_provider.providers.openai_provider.asyncio.sleep") as sleep_mock:
                    async for chunk in provider.stream(
                        model,
                        [{"role": "user", "content": "hi"}],
                        options,
                        api_key="test-key",
                    ):
                        chunks.append(chunk)
                    sleep_mock.assert_awaited_once()
                self.assertEqual(calls["count"], 2)
                return chunks

        chunks = asyncio.run(collect())
        self.assertEqual(chunks, [{"content": "hello"}])

    def test_stream_rejects_retry_delay_above_cap(self):
        model = Model(id="gpt-5.4", provider="openai", api="openai-chat", maxTokens=1024)
        provider = OpenAiProvider()
        options = StreamOptions(max_retry_delay_ms=500)

        async def collect():
            with patch("openai.AsyncOpenAI") as mock_client_cls:
                request = httpx.Request("POST", "https://example.com/v1/chat/completions")
                response = httpx.Response(429, headers={"Retry-After": "1"}, request=request)
                rate_limit_error = RateLimitError(
                    "Please retry in 1s",
                    response=response,
                    body={"error": {"message": "Please retry in 1s"}},
                )

                async def fake_create(**kwargs):
                    raise rate_limit_error

                mock_client = mock_client_cls.return_value
                mock_client.chat.completions.create.side_effect = fake_create

                async for _chunk in provider.stream(
                    model,
                    [{"role": "user", "content": "hi"}],
                    options,
                    api_key="test-key",
                ):
                    pass

        with self.assertRaisesRegex(ValueError, "max_retry_delay_ms=500"):
            asyncio.run(collect())

    def test_stream_normalizes_messages_and_uses_strict_tools(self):
        model = Model(
            id="compat-openai",
            provider="openai-compatible",
            api="openai-chat",
            maxTokens=1024,
            compat=ModelCompat(
                supportsDeveloperRole=False,
                supportsStrictToolSchema=True,
                requiresToolResultName=True,
                requiresAssistantAfterToolResult=True,
            ),
        )
        provider = OpenAiProvider()
        options = StreamOptions(
            tools=[
                AgentTool(
                    name="analyze_dataframe",
                    description="Analyze a dataframe-like CSV.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                    label="Analyze Dataframe",
                )
            ]
        )

        class FakeStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        async def collect():
            with patch("openai.AsyncOpenAI") as mock_client_cls:
                captured = {}

                async def fake_create(**kwargs):
                    captured.update(kwargs)
                    return FakeStream()

                mock_client = mock_client_cls.return_value
                mock_client.chat.completions.create.side_effect = fake_create

                messages = [
                    {"role": "developer", "content": "Prefer code."},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "tool_call_id": "call_1", "tool_name": "python", "content": "done"},
                    {"role": "user", "content": "continue"},
                ]

                chunks = []
                async for chunk in provider.stream(model, messages, options, api_key="test-key"):
                    chunks.append(chunk)
                return captured, chunks

        captured, chunks = asyncio.run(collect())

        self.assertEqual(chunks, [])
        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][2]["name"], "python")
        self.assertEqual(captured["messages"][3], {"role": "assistant", "content": ""})
        self.assertEqual(captured["messages"][4]["role"], "user")
        self.assertEqual(captured["tools"][0]["function"]["strict"], True)
        self.assertEqual(captured["tools"][0]["function"]["parameters"]["additionalProperties"], False)


if __name__ == "__main__":
    unittest.main()
