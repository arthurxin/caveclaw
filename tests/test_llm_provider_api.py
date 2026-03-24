import asyncio
import os
import unittest
from unittest.mock import patch

from agent_core import Message
from agent_core.llm_provider import (
    CostConfig,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    api_provider_registry,
    calculate_usage_cost,
    complete,
    complete_simple,
    finalize_usage_payload,
    get_env_api_key,
    prepare_provider_stream,
    register_builtin_providers,
    stream,
    stream_simple,
)


class FakeLowLevelProvider:
    api = "fake-low-level-provider"

    def __init__(self):
        self.last_options = None
        self.last_api_key = None
        self.last_messages = None

    async def stream(self, model, messages, options, api_key=None):
        self.last_options = options
        self.last_api_key = api_key
        self.last_messages = messages
        yield {"content": "hello"}
        yield {"usage": {"input_tokens": 100, "output_tokens": 50}}


class FakePayloadCapturingProvider:
    api = "fake-payload-capturing-provider"

    def __init__(self):
        self.last_payload = None

    async def stream(self, model, messages, options, api_key=None):
        self.last_payload = dict(messages=messages, session_id=options.session_id, metadata=options.metadata)
        yield {"content": "payload captured"}


class LlmProviderApiTests(unittest.TestCase):
    def setUp(self):
        self._original_providers = api_provider_registry.list()
        api_provider_registry.clear()

    def tearDown(self):
        api_provider_registry.clear()
        for provider in self._original_providers:
            api_provider_registry.register(provider)

    def test_get_env_api_key_uses_standard_provider_naming(self):
        with patch.dict(os.environ, {"FAKE_PROVIDER_API_KEY": "env-secret"}, clear=False):
            self.assertEqual(get_env_api_key("fake-provider"), "env-secret")

    def test_register_builtin_providers_populates_registry(self):
        register_builtin_providers()

        self.assertIsNotNone(api_provider_registry.get("openai-chat"))
        self.assertIsNotNone(api_provider_registry.get("azure-responses"))
        self.assertIsNotNone(api_provider_registry.get("google-gemini"))

    def test_complete_uses_low_level_provider_substrate(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)
        options = StreamOptions(
            api_key="direct-key",
            system_prompt="sys",
            temperature=0.3,
            max_tokens=111,
            session_id="session-low-level",
            metadata={"trace_id": "t-1"},
        )

        assistant = asyncio.run(complete(model, [Message(role="user", content="hello")], options=options))

        self.assertEqual(assistant.content, "hello")
        self.assertEqual(provider.last_api_key, "direct-key")
        self.assertEqual(provider.last_options.system_prompt, "sys")
        self.assertEqual(provider.last_options.temperature, 0.3)
        self.assertEqual(provider.last_options.max_tokens, 111)
        self.assertEqual(provider.last_options.session_id, "session-low-level")
        self.assertEqual(provider.last_options.metadata, {"trace_id": "t-1"})

    def test_stream_yields_normalized_chunks(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)

        async def collect():
            chunks = []
            async for chunk in stream(model, [Message(role="user", content="hello")], options=StreamOptions(api_key="k")):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())

        self.assertEqual(chunks, [{"content": "hello"}, {"usage": {"input_tokens": 100, "output_tokens": 50}}])

    def test_prepare_provider_stream_returns_codec_and_chunk_stream(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)

        prepared = prepare_provider_stream(
            model,
            [Message(role="user", content="hello")],
            options=StreamOptions(api_key="k"),
        )

        self.assertIsNotNone(prepared.message_codec)
        self.assertEqual(prepared.partial_message.model, "fake-model")

    def test_complete_simple_maps_reasoning_to_thinking_level(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)

        assistant = asyncio.run(
            complete_simple(
                model,
                [Message(role="user", content="hello")],
                options=SimpleStreamOptions(api_key="simple-key", reasoning="high"),
            )
        )

        self.assertEqual(assistant.content, "hello")
        self.assertEqual(provider.last_options.thinking_level, "high")

    def test_stream_simple_yields_chunks(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)

        async def collect():
            chunks = []
            async for chunk in stream_simple(
                model,
                [Message(role="user", content="hello")],
                options=SimpleStreamOptions(api_key="k", reasoning="medium"),
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())

        self.assertEqual(chunks, [{"content": "hello"}, {"usage": {"input_tokens": 100, "output_tokens": 50}}])
        self.assertEqual(provider.last_options.thinking_level, "medium")

    def test_complete_finalizes_usage_and_cost(self):
        provider = FakeLowLevelProvider()
        api_provider_registry.register(provider)
        model = Model(
            id="fake-model",
            provider="fake-provider",
            api=provider.api,
            cost=CostConfig(input=2_000.0, output=4_000.0),
        )

        assistant = asyncio.run(complete(model, [Message(role="user", content="hello")], options=StreamOptions(api_key="k")))

        self.assertEqual(assistant.usage["input"], 100)
        self.assertEqual(assistant.usage["output"], 50)
        self.assertEqual(assistant.usage["totalTokens"], 150)
        self.assertEqual(assistant.usage["cost"]["input"], 0.2)
        self.assertEqual(assistant.usage["cost"]["output"], 0.2)
        self.assertEqual(assistant.usage["cost"]["total"], 0.4)

    def test_finalize_usage_payload_preserves_extra_fields(self):
        usage = finalize_usage_payload(
            {"input_tokens": 10, "output_tokens": 5, "thoughts_token_count": 3},
            None,
        )

        self.assertEqual(usage["input"], 10)
        self.assertEqual(usage["output"], 5)
        self.assertEqual(usage["totalTokens"], 15)
        self.assertEqual(usage["thoughts_token_count"], 3)
        self.assertEqual(usage["cost"]["total"], 0.0)

    def test_calculate_usage_cost_uses_model_rates(self):
        model = Model(
            id="fake-model",
            provider="fake-provider",
            api="fake-api",
            cost=CostConfig(input=1_500.0, output=500.0, cacheRead=100.0, cacheWrite=200.0),
        )

        cost = calculate_usage_cost(
            model,
            {"input": 200, "output": 100, "cacheRead": 50, "cacheWrite": 25},
        )

        self.assertAlmostEqual(cost["input"], 0.3)
        self.assertAlmostEqual(cost["output"], 0.05)
        self.assertAlmostEqual(cost["cacheRead"], 0.005)
        self.assertAlmostEqual(cost["cacheWrite"], 0.005)
        self.assertAlmostEqual(cost["total"], 0.36)

    def test_prepare_provider_stream_keeps_session_id_available_for_providers(self):
        provider = FakePayloadCapturingProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake-provider", api=provider.api)

        asyncio.run(
            complete(
                model,
                [Message(role="user", content="hello")],
                options=StreamOptions(api_key="k", session_id="session-123", metadata={"trace_id": "trace-1"}),
            )
        )

        self.assertEqual(provider.last_payload["session_id"], "session-123")
        self.assertEqual(provider.last_payload["metadata"], {"trace_id": "trace-1"})


if __name__ == "__main__":
    unittest.main()
