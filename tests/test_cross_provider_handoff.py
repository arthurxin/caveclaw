import unittest

from agent_core import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultMessage,
    handoff_messages,
)
from agent_core.assistant_messages import AssistantMessage, RuntimeSnapshotBlock, RuntimeSnapshotEntry
from agent_core.llm_provider import Model
from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.message_codec import AnthropicMessageCodec, AzureMessageCodec, GoogleMessageCodec
from agent_core.llm_provider.providers.azure_provider import _messages_to_azure_input
from agent_core.llm_provider.providers.google_provider import _convert_messages


def _openai_source_model() -> Model:
    return Model(id="gpt-5.4", provider="openai", api="openai-chat")


def _google_source_model() -> Model:
    return Model(id="gemini-3.1-pro-preview", provider="google", api="google-gemini")


def _azure_source_model() -> Model:
    return Model(id="gpt-5.4", provider="azure", api="azure-responses")


def _anthropic_target_model() -> Model:
    return Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")


def _google_target_model() -> Model:
    return Model(id="gemini-3.1-pro-preview", provider="google", api="google-gemini")


def _azure_target_model() -> Model:
    return Model(id="gpt-5.4", provider="azure", api="azure-responses")


class CrossProviderHandoffTests(unittest.TestCase):
    def test_openai_style_context_can_handoff_to_anthropic_codec(self):
        source_model = _openai_source_model()
        target_model = _anthropic_target_model()
        messages = [
            Message(role="user", content="Please look up Beijing weather."),
            AssistantMessage(
                content_blocks=[
                    ThinkingBlock(thinking="Need to call weather tool."),
                    RuntimeSnapshotBlock(
                        entries=[
                            RuntimeSnapshotEntry(
                                key="dataset",
                                version=3,
                                summary_blocks=[TextBlock(text="weather cache is warm")],
                            )
                        ]
                    ),
                    ToolCallBlock(id="call|weather", name="getWeather", arguments={"city": "Beijing"}),
                ],
                tool_calls=[ToolCall(id="call|weather", name="getWeather", arguments={"city": "Beijing"})],
                provider_state={"openai": {"response_id": "resp_openai_1"}},
                raw_content="<think>Need to call weather tool.</think>\ncalling tool",
                stop_reason="tool_use",
                model=source_model.id,
                provider=source_model.provider,
                api=source_model.api,
            ),
            ToolResultMessage(tool_call_id="call|weather", name="getWeather", content="Sunny, 31C"),
            AssistantMessage(
                content_blocks=[TextBlock(text="It is sunny and 31C.")],
                model=source_model.id,
                provider=source_model.provider,
                api=source_model.api,
            ),
        ]

        result = handoff_messages(messages, target_model)
        codec = AnthropicMessageCodec()
        payloads = codec.encode_messages(result.messages, StreamOptions())

        self.assertEqual(payloads[1]["role"], "assistant")
        assistant_content = payloads[1]["content"]
        self.assertEqual(assistant_content[0]["type"], "text")
        self.assertIn("Runtime Snapshot:", assistant_content[0]["text"])
        self.assertEqual(assistant_content[1]["type"], "tool_use")
        self.assertEqual(payloads[2]["role"], "user")
        self.assertEqual(payloads[2]["content"][0]["type"], "tool_result")
        self.assertTrue(any(d.code == "tool_call_id_rewritten" for d in result.diagnostics))
        self.assertTrue(any(d.code == "runtime_block_converted" for d in result.diagnostics))

    def test_google_style_context_can_handoff_to_azure_input(self):
        source_model = _google_source_model()
        target_model = _azure_target_model()
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="Use provider-native gemini parts."),
                ToolCallBlock(id="gemini_call_1", name="lookup_stock", arguments={"ticker": "AAPL"}),
            ],
            tool_calls=[ToolCall(id="gemini_call_1", name="lookup_stock", arguments={"ticker": "AAPL"})],
            provider_state={
                "gemini": {
                    "parts": [
                        {"text": "Use provider-native gemini parts.", "thoughtSignature": "sig-1"},
                        {"functionCall": {"name": "lookup_stock", "args": {"ticker": "AAPL"}}},
                    ],
                    "thought_signatures": ["sig-1"],
                }
            },
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
            stop_reason="tool_use",
        )
        messages = [
            Message(role="user", content="Check Apple stock."),
            assistant,
            ToolResultMessage(tool_call_id="gemini_call_1", name="lookup_stock", content="188.20"),
        ]

        result = handoff_messages(messages, target_model)
        codec = AzureMessageCodec()
        payloads = codec.encode_messages(result.messages, StreamOptions())
        azure_items = _messages_to_azure_input(payloads, allow_system=False)

        self.assertTrue(result.rewind_applied)
        self.assertEqual(len(azure_items), 1)
        self.assertEqual(azure_items[0]["role"], "user")
        self.assertTrue(any(d.code == "rewind_applied" for d in result.diagnostics))

    def test_azure_style_context_can_handoff_to_google_contents(self):
        source_model = _azure_source_model()
        target_model = _google_target_model()
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="Need to search flights."),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="travel_plan",
                            version=7,
                            summary_blocks=[TextBlock(text="destination=Tokyo, dates=next week")],
                        )
                    ]
                ),
                ToolCallBlock(id="azure_call_1", name="search_flights", arguments={"to": "Tokyo"}),
            ],
            tool_calls=[ToolCall(id="azure_call_1", name="search_flights", arguments={"to": "Tokyo"})],
            provider_state={"azure": {"response_id": "resp_azure_1", "status": "completed"}},
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
            stop_reason="tool_use",
        )
        messages = [
            Message(role="user", content="Find flights to Tokyo."),
            assistant,
            ToolResultMessage(tool_call_id="azure_call_1", name="search_flights", content="3 flights found"),
            Message(role="user", content="continue planning"),
        ]

        result = handoff_messages(messages, target_model)
        codec = GoogleMessageCodec()
        payloads = codec.encode_messages(result.messages, StreamOptions())
        history, last_user = _convert_messages(payloads)

        self.assertTrue(result.rewind_applied)
        self.assertEqual(history, [])
        self.assertEqual(last_user[0], "user")
        self.assertEqual(last_user[1][0]["text"], "Find flights to Tokyo.")
        self.assertTrue(any(d.code == "rewind_applied" for d in result.diagnostics))


if __name__ == "__main__":
    unittest.main()
