import unittest

from agent_core.assistant_messages import AssistantMessage, ToolCall, ToolResultMessage
from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.message_codec import AzureMessageCodec
from agent_core.llm_provider.provider_types import Model
from agent_core.llm_provider.providers.azure_provider import (
    _build_azure_payload,
    _find_previous_response_context,
    _parse_azure_response,
)


class AzureProviderTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
