import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import RateLimitError

from agent_core.assistant_messages import AssistantMessage, ToolCall, ToolResultMessage
from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.message_codec import AzureMessageCodec
from agent_core.llm_provider.provider_types import Model
from agent_core.llm_provider.providers.azure_provider import (
    AzureProvider,
    _build_azure_payload,
    _extract_azure_base_url,
    _find_previous_response_context,
    _parse_azure_response,
)


class AzureProviderTests(unittest.TestCase):
    def test_extract_azure_base_url_strips_responses_suffix(self):
        self.assertEqual(
            _extract_azure_base_url("https://example.openai.azure.com/openai/v1/responses"),
            "https://example.openai.azure.com/openai/v1",
        )

    def test_build_azure_payload_matches_responses_shape(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        options = StreamOptions()
        options.system_prompt = "你是一个专业且幽默的助手。"
        options.thinking_level = "medium"

        payload = _build_azure_payload(
            [
                {"role": "user", "content": "你好, 请问你是?"},
            ],
            model,
            options,
        )

        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertEqual(payload["instructions"], "你是一个专业且幽默的助手。")
        self.assertEqual(payload["input"][0]["role"], "user")
        self.assertEqual(payload["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(payload["reasoning"]["effort"], "medium")

    def test_build_azure_payload_honors_stream_option_overrides(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        options = StreamOptions()
        options.max_tokens = 321
        options.temperature = 0.15
        options.metadata = {"trace_id": "trace-1"}

        payload = _build_azure_payload(
            [{"role": "user", "content": "你好"}],
            model,
            options,
        )

        self.assertEqual(payload["max_output_tokens"], 321)
        self.assertEqual(payload["temperature"], 0.15)
        self.assertEqual(payload["metadata"], {"trace_id": "trace-1"})

    def test_build_azure_payload_injects_session_id_into_metadata(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        options = StreamOptions()
        options.session_id = "session-xyz"
        options.metadata = {"trace_id": "trace-1"}

        payload = _build_azure_payload(
            [{"role": "user", "content": "你好"}],
            model,
            options,
        )

        self.assertEqual(payload["metadata"]["trace_id"], "trace-1")
        self.assertEqual(payload["metadata"]["session_id"], "session-xyz")

    def test_build_azure_payload_includes_tools_and_previous_response_id(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        options = StreamOptions()
        options.tools = [
            type(
                "Tool",
                (),
                {
                    "name": "getWeather",
                    "description": "获得指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            )()
        ]

        assistant = AssistantMessage(
            content_blocks=[],
            tool_calls=[ToolCall(id="call_1", name="getWeather", arguments={"city": "北京"})],
            provider_state={"azure": {"response_id": "resp_123"}},
        )
        tool_result = ToolResultMessage(tool_call_id="call_1", name="getWeather", content="晴天, 31°C")

        codec = AzureMessageCodec()
        provider_messages = codec.to_provider_messages([assistant, tool_result], options)
        payload = _build_azure_payload(provider_messages, model, options)

        self.assertEqual(payload["previous_response_id"], "resp_123")
        self.assertEqual(payload["input"][0]["type"], "function_call_output")
        self.assertEqual(payload["input"][0]["call_id"], "call_1")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["name"], "getWeather")
        self.assertTrue(payload["parallel_tool_calls"])

    def test_parse_azure_response_extracts_reasoning_text_and_tool_calls(self):
        provider_state, reasoning, content, tool_calls = _parse_azure_response(
            {
                "id": "resp_456",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"text": "先查天气，再做加法。"},
                        ],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "getWeather",
                        "arguments": "{\"city\":\"北京\"}",
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "你好，我是 Azure 上的 GPT-5.4。",
                            }
                        ],
                    },
                ]
            }
        )

        self.assertEqual(provider_state["response_id"], "resp_456")
        self.assertEqual(provider_state["status"], "completed")
        self.assertEqual(reasoning, "先查天气，再做加法。")
        self.assertEqual(content, "你好，我是 Azure 上的 GPT-5.4。")
        self.assertEqual(tool_calls[0]["id"], "call_1")
        self.assertEqual(tool_calls[0]["name"], "getWeather")
        self.assertEqual(tool_calls[0]["arguments"]["city"], "北京")

    def test_find_previous_response_context_uses_latest_azure_assistant(self):
        previous_response_id, tail_messages = _find_previous_response_context(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi", "provider_state": {"azure": {"response_id": "resp_1"}}},
                {"role": "tool", "tool_call_id": "call_1", "name": "getWeather", "content": "晴天"},
            ]
        )

        self.assertEqual(previous_response_id, "resp_1")
        self.assertEqual(len(tail_messages), 1)
        self.assertEqual(tail_messages[0]["tool_call_id"], "call_1")

    def test_stream_normalizes_azure_response_events(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        provider = AzureProvider()
        options = StreamOptions()

        async def collect():
            with patch("openai.AsyncOpenAI") as mock_client_cls:
                stream_events = [
                    SimpleNamespace(type="response.reasoning_summary_text.delta", delta="先查天气"),
                    SimpleNamespace(type="response.output_text.delta", delta="你好"),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(
                            type="function_call",
                            call_id="call_1",
                            id="fc_1",
                            name="getWeather",
                            arguments='{"city":"北京"}',
                        ),
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            id="resp_789",
                            status="completed",
                            usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
                        ),
                    ),
                ]

                class FakeStream:
                    def __init__(self, events):
                        self._events = events

                    def __aiter__(self):
                        self._iterator = iter(self._events)
                        return self

                    async def __anext__(self):
                        try:
                            return next(self._iterator)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc

                async def fake_create(**kwargs):
                    self.assertEqual(kwargs["model"], "gpt-5.4")
                    self.assertTrue(kwargs["stream"])
                    return FakeStream(stream_events)

                mock_client = mock_client_cls.return_value
                mock_client.responses.create.side_effect = fake_create

                output = []
                async for chunk in provider.stream(
                    model,
                    [{"role": "user", "content": "你好"}],
                    options,
                    api_key="test-key",
                ):
                    output.append(chunk)
                return output

        import asyncio

        chunks = asyncio.run(collect())

        self.assertEqual(chunks[0]["reasoning"], "先查天气")
        self.assertEqual(chunks[1]["content"], "你好")
        self.assertEqual(chunks[2]["tool_calls"][0]["arguments"]["city"], "北京")
        self.assertEqual(chunks[3]["provider_state"]["azure"]["response_id"], "resp_789")
        self.assertEqual(chunks[4]["usage"]["total_tokens"], 18)

    def test_stream_retries_when_delay_is_within_cap(self):
        model = Model(
            id="gpt-5.4",
            provider="azure",
            api="azure-responses",
            baseUrl="https://example.openai.azure.com/openai/v1/responses",
            reasoning=True,
            maxTokens=16384,
        )
        provider = AzureProvider()
        options = StreamOptions(max_retry_delay_ms=1500)

        async def collect():
            with patch("openai.AsyncOpenAI") as mock_client_cls:
                request = httpx.Request("POST", "https://example.openai.azure.com/openai/v1/responses")
                response = httpx.Response(429, headers={"Retry-After": "0"}, request=request)
                rate_limit_error = RateLimitError(
                    "Please retry in 0s",
                    response=response,
                    body={"error": {"message": "Please retry in 0s"}},
                )

                stream_events = [
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            id="resp_retry",
                            status="completed",
                            usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
                        ),
                    ),
                ]

                class FakeStream:
                    def __init__(self, events):
                        self._events = events

                    def __aiter__(self):
                        self._iterator = iter(self._events)
                        return self

                    async def __anext__(self):
                        try:
                            return next(self._iterator)
                        except StopIteration as exc:
                            raise StopAsyncIteration from exc

                calls = {"count": 0}

                async def fake_create(**kwargs):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise rate_limit_error
                    return FakeStream(stream_events)

                mock_client = mock_client_cls.return_value
                mock_client.responses.create.side_effect = fake_create

                with patch("agent_core.llm_provider.providers.azure_provider.asyncio.sleep") as sleep_mock:
                    output = []
                    async for chunk in provider.stream(
                        model,
                        [{"role": "user", "content": "你好"}],
                        options,
                        api_key="test-key",
                    ):
                        output.append(chunk)
                    sleep_mock.assert_awaited_once()
                self.assertEqual(calls["count"], 2)
                return output

        import asyncio

        chunks = asyncio.run(collect())
        self.assertEqual(chunks[0]["provider_state"]["azure"]["response_id"], "resp_retry")
        self.assertEqual(chunks[1]["usage"]["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
