import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.provider_types import Model
from agent_core.llm_provider.providers.anthropic_provider import AnthropicProvider


class AnthropicProviderTests(unittest.TestCase):
    def test_stream_emits_reasoning_provider_state_tool_calls_and_usage(self):
        model = Model(
            id="claude-test",
            provider="anthropic",
            api="anthropic-messages",
            reasoning=True,
            maxTokens=1024,
        )
        provider = AnthropicProvider()
        options = StreamOptions()
        options.thinking_level = "high"
        options.thinking_budgets = {"high": 1234}

        class FakeStreamContext:
            def __init__(self, events):
                self._events = events

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                self._iterator = iter(self._events)
                return self

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration as exc:
                    raise StopAsyncIteration from exc

        async def collect():
            captured = {}
            fake_client = SimpleNamespace()

            def fake_stream(**kwargs):
                captured.update(kwargs)
                return FakeStreamContext(
                    [
                        SimpleNamespace(
                            type="content_block_start",
                            content_block=SimpleNamespace(type="thinking"),
                        ),
                        SimpleNamespace(
                            type="content_block_delta",
                            delta=SimpleNamespace(type="thinking_delta", thinking="plan"),
                        ),
                        SimpleNamespace(
                            type="content_block_delta",
                            delta=SimpleNamespace(type="signature_delta", signature="sig-1"),
                        ),
                        SimpleNamespace(
                            type="content_block_stop",
                        ),
                        SimpleNamespace(
                            type="content_block_start",
                            content_block=SimpleNamespace(type="tool_use", id="tool_1", name="lookup"),
                        ),
                        SimpleNamespace(
                            type="content_block_delta",
                            delta=SimpleNamespace(type="input_json_delta", partial_json='{"city":"Beijing"}'),
                        ),
                        SimpleNamespace(type="content_block_stop"),
                        SimpleNamespace(
                            type="message_delta",
                            usage=SimpleNamespace(
                                input_tokens=10,
                                output_tokens=5,
                                cache_read_input_tokens=2,
                                cache_creation_input_tokens=3,
                            ),
                        ),
                    ]
                )

            fake_client.messages = SimpleNamespace(stream=fake_stream)
            fake_module = SimpleNamespace(AsyncAnthropic=lambda **kwargs: fake_client)

            with patch.dict(sys.modules, {"anthropic": fake_module}):
                chunks = []
                async for chunk in provider.stream(
                    model,
                    [{"role": "user", "content": "hello"}],
                    options,
                    api_key="test-key",
                ):
                    chunks.append(chunk)
            return captured, chunks

        captured, chunks = asyncio.run(collect())

        self.assertEqual(captured["thinking"], {"type": "enabled", "budget_tokens": 1234})
        self.assertEqual(chunks[0], {"reasoning": "plan"})
        self.assertEqual(chunks[1], {"provider_state": {"anthropic": {"thought_signatures": ["sig-1"]}}})
        self.assertEqual(chunks[2]["tool_calls"][0]["name"], "lookup")
        self.assertEqual(chunks[2]["tool_calls"][0]["arguments"]["city"], "Beijing")
        self.assertEqual(chunks[3]["usage"]["input"], 10)
        self.assertEqual(chunks[3]["usage"]["output"], 5)
        self.assertEqual(chunks[3]["usage"]["cacheRead"], 2)
        self.assertEqual(chunks[3]["usage"]["cacheWrite"], 3)


if __name__ == "__main__":
    unittest.main()
