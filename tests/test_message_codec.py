import unittest

from agent_core.assistant_stream import append_assistant_delta
from agent_core.llm.api_registry import StreamOptions
from agent_core.llm.message_codec import AnthropicMessageCodec, ArkMessageCodec, AzureMessageCodec, GoogleMessageCodec, MiniMaxMessageCodec
from agent_core.llm.providers.google_provider import (
    _convert_messages,
    _convert_schema_to_gemini,
    _extract_gemini_provider_state,
)
from agent_core.types import AssistantMessage, Message, ToolCall, ToolResultMessage


class MessageCodecTests(unittest.TestCase):
    def test_minimax_codec_prefers_raw_content_for_replay(self):
        codec = MiniMaxMessageCodec()
        message = Message(role="assistant", content="clean", raw_content="<think>plan</think>\nclean")

        payloads = codec.to_provider_messages([message], StreamOptions())

        self.assertEqual(payloads[0]["content"], "<think>plan</think>\nclean")

    def test_google_codec_carries_namespaced_provider_state(self):
        codec = GoogleMessageCodec()
        message = Message(role="assistant", content="hello")
        message.set_provider_state("gemini", {"parts": [{"text": "hello", "thoughtSignature": "sig-1"}]})

        payloads = codec.to_provider_messages([message], StreamOptions())

        self.assertIn("provider_state", payloads[0])
        self.assertIn("gemini", payloads[0]["provider_state"])
        self.assertEqual(payloads[0]["provider_state"]["gemini"]["parts"][0]["thoughtSignature"], "sig-1")

    def test_google_convert_messages_prefers_provider_parts(self):
        messages = [
            {
                "role": "assistant",
                "content": "ignored",
                "provider_state": {
                    "gemini": {
                        "parts": [{"text": "preserved", "thoughtSignature": "sig-2"}],
                    }
                },
            },
            {"role": "user", "content": "next"},
        ]

        history, last_user = _convert_messages(messages)

        self.assertEqual(last_user, ("user", [{"text": "next"}]))
        self.assertEqual(history[0]["parts"][0]["text"], "preserved")
        self.assertEqual(history[0]["parts"][0]["thoughtSignature"], "sig-2")

    def test_google_convert_messages_merges_consecutive_tool_results(self):
        messages = [
            {"role": "user", "content": "帮我查天气"},
            {
                "role": "assistant",
                "provider_state": {
                    "gemini": {
                        "parts": [
                            {"functionCall": {"name": "getWeather", "args": {"city": "北京"}}},
                            {"functionCall": {"name": "getWeather", "args": {"city": "上海"}}},
                        ]
                    }
                },
            },
            {
                "role": "tool",
                "name": "getWeather",
                "content": "晴天, 31°C",
                "provider_state": {
                    "gemini": {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "getWeather",
                                    "response": {"result": "晴天, 31°C"},
                                }
                            }
                        ]
                    }
                },
            },
            {
                "role": "tool",
                "name": "getWeather",
                "content": "多云, 32°C",
                "provider_state": {
                    "gemini": {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "getWeather",
                                    "response": {"result": "多云, 32°C"},
                                }
                            }
                        ]
                    }
                },
            },
        ]

        history, last_user = _convert_messages(messages)

        self.assertEqual(len(history), 2)
        self.assertEqual(last_user[0], "user")
        self.assertEqual(len(last_user[1]), 2)
        self.assertEqual(last_user[1][0]["functionResponse"]["name"], "getWeather")
        self.assertEqual(last_user[1][1]["functionResponse"]["response"]["result"], "多云, 32°C")

    def test_assistant_delta_merges_provider_state_lists(self):
        message = AssistantMessage(content_blocks=[])

        append_assistant_delta(message, {"provider_state": {"gemini": {"thought_signatures": ["sig-1"]}}})
        append_assistant_delta(message, {"provider_state": {"gemini": {"thought_signatures": ["sig-2"]}}})

        self.assertEqual(message.provider_state["gemini"]["thought_signatures"], ["sig-1", "sig-2"])

    def test_google_codec_infers_function_call_and_tool_response_parts(self):
        codec = GoogleMessageCodec()
        assistant = AssistantMessage(content_blocks=[], tool_calls=[ToolCall(id="call_1", name="getWeather", arguments={"city": "北京"})])
        tool_result = ToolResultMessage(tool_call_id="call_1", name="getWeather", content="晴天, 31°C")

        payloads = codec.to_provider_messages([assistant, tool_result], StreamOptions())

        self.assertEqual(payloads[0]["provider_state"]["gemini"]["parts"][0]["functionCall"]["name"], "getWeather")
        self.assertEqual(payloads[1]["provider_state"]["gemini"]["parts"][0]["functionResponse"]["name"], "getWeather")

    def test_anthropic_codec_maps_tool_history(self):
        codec = AnthropicMessageCodec()
        assistant = AssistantMessage(content_blocks=[], tool_calls=[ToolCall(id="call_a", name="getSum", arguments={"numbers": [1, 2]})])
        tool_result = ToolResultMessage(tool_call_id="call_a", name="getSum", content="3")

        payloads = codec.to_provider_messages([assistant, tool_result], StreamOptions())

        self.assertEqual(payloads[0]["role"], "assistant")
        self.assertEqual(payloads[0]["content"][0]["type"], "tool_use")
        self.assertEqual(payloads[1]["role"], "user")
        self.assertEqual(payloads[1]["content"][0]["type"], "tool_result")

    def test_minimax_codec_from_provider_chunk_stores_replay_payload(self):
        codec = MiniMaxMessageCodec()
        assistant = AssistantMessage(content_blocks=[])

        normalized = codec.from_provider_chunk(
            {"raw_content": "<think>plan</think>\nanswer", "content": "answer"},
            assistant,
        )

        self.assertEqual(normalized["provider_state"]["minimax"]["replay_payload"]["content"], "<think>plan</think>\nanswer")

    def test_google_codec_finalize_deduplicates_thought_signatures(self):
        codec = GoogleMessageCodec()
        assistant = AssistantMessage(content_blocks=[])
        assistant.set_provider_state("gemini", {"thought_signatures": ["sig-1", "sig-1", "sig-2"]})

        codec.finalize_assistant_message(assistant)

        self.assertEqual(assistant.provider_state["gemini"]["thought_signatures"], ["sig-1", "sig-2"])

    def test_ark_codec_stringifies_tool_call_arguments_for_replay(self):
        codec = ArkMessageCodec()
        assistant = AssistantMessage(content_blocks=[], tool_calls=[ToolCall(id="call_ark", name="getWeather", arguments={"city": "北京"})])

        payloads = codec.to_provider_messages([assistant], StreamOptions())

        self.assertEqual(payloads[0]["tool_calls"][0]["function"]["arguments"], '{"city": "北京"}')

    def test_azure_codec_preserves_provider_state_for_replay(self):
        codec = AzureMessageCodec()
        assistant = AssistantMessage(content_blocks=[], provider_state={"azure": {"response_id": "resp_123"}})

        payloads = codec.to_provider_messages([assistant], StreamOptions())

        self.assertEqual(payloads[0]["provider_state"]["azure"]["response_id"], "resp_123")

    def test_google_schema_converter_maps_object_type(self):
        schema = _convert_schema_to_gemini(
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        )

        self.assertEqual(schema["type"], "OBJECT")
        self.assertEqual(schema["properties"]["city"]["type"], "STRING")

    def test_google_provider_state_extracts_thought_signature_from_function_call_part(self):
        provider_state = _extract_gemini_provider_state(
            {
                "functionCall": {
                    "name": "getWeather",
                    "args": {"city": "北京"},
                },
                "thoughtSignature": "sig-1",
            }
        )

        self.assertEqual(provider_state["thought_signatures"], ["sig-1"])
        self.assertEqual(provider_state["parts"][0]["functionCall"]["name"], "getWeather")
        self.assertEqual(provider_state["parts"][0]["thoughtSignature"], "sig-1")


if __name__ == "__main__":
    unittest.main()
