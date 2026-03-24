import unittest
from typing import Callable, Dict, List, Tuple

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
from agent_core.llm_provider import Model, normalize_openai_compatible_messages
from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.codecs.providers import ArkMessageCodec, DefaultMessageCodec, MiniMaxMessageCodec
from agent_core.llm_provider.message_codec import AnthropicMessageCodec, AzureMessageCodec, GoogleMessageCodec
from agent_core.llm_provider.providers.azure_provider import _messages_to_azure_input
from agent_core.llm_provider.providers.google_provider import _convert_messages


SourceFactory = Callable[[], List[Message]]
TargetValidator = Callable[[unittest.TestCase, Model, List[Message]], None]


def _source_openai_tool_history() -> List[Message]:
    model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
    return [
        Message(role="user", content="Look up Beijing weather."),
        AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="Need to call getWeather."),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="weather_cache",
                            version=2,
                            summary_blocks=[TextBlock(text="cache_status=warm")],
                        )
                    ]
                ),
                ToolCallBlock(id="call|weather", name="getWeather", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|weather", name="getWeather", arguments={"city": "Beijing"})],
            provider_state={"openai": {"response_id": "resp_openai_1"}},
            raw_content="<think>Need to call getWeather.</think>\ncalling tool",
            stop_reason="tool_use",
            model=model.id,
            provider=model.provider,
            api=model.api,
        ),
        ToolResultMessage(tool_call_id="call|weather", name="getWeather", content="Sunny, 31C"),
        AssistantMessage(
            content_blocks=[TextBlock(text="It is sunny and 31C.")],
            model=model.id,
            provider=model.provider,
            api=model.api,
        ),
    ]


def _source_google_provider_state() -> List[Message]:
    model = Model(id="gemini-3.1-pro-preview", provider="google", api="google-gemini")
    return [
        Message(role="user", content="Check Apple stock."),
        AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="Use Gemini-native parts."),
                ToolCallBlock(id="gemini_call_1", name="lookup_stock", arguments={"ticker": "AAPL"}),
            ],
            tool_calls=[ToolCall(id="gemini_call_1", name="lookup_stock", arguments={"ticker": "AAPL"})],
            provider_state={
                "gemini": {
                    "parts": [
                        {"text": "Use Gemini-native parts.", "thoughtSignature": "sig-1"},
                        {"functionCall": {"name": "lookup_stock", "args": {"ticker": "AAPL"}}},
                    ],
                    "thought_signatures": ["sig-1"],
                }
            },
            stop_reason="tool_use",
            model=model.id,
            provider=model.provider,
            api=model.api,
        ),
        ToolResultMessage(tool_call_id="gemini_call_1", name="lookup_stock", content="188.20"),
    ]


def _source_azure_continuation() -> List[Message]:
    model = Model(id="gpt-5.4", provider="azure", api="azure-responses")
    return [
        Message(role="user", content="Find flights to Tokyo."),
        AssistantMessage(
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
            stop_reason="tool_use",
            model=model.id,
            provider=model.provider,
            api=model.api,
        ),
        ToolResultMessage(tool_call_id="azure_call_1", name="search_flights", content="3 flights found"),
        Message(role="user", content="continue planning"),
    ]


def _source_minimax_reasoning() -> List[Message]:
    model = Model(id="minimax", provider="minimax", api="minimax-local")
    return [
        Message(role="user", content="Summarize latest notes."),
        AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="First inspect runtime summary."),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="notes",
                            version=5,
                            summary_blocks=[TextBlock(text="3 bullet points prepared")],
                        )
                    ]
                ),
                TextBlock(text="Ready to summarize."),
            ],
            raw_content="<think>First inspect runtime summary.</think>\nReady to summarize.",
            provider_state={"minimax": {"replay_payload": {"role": "assistant", "content": "<think>First inspect runtime summary.</think>\nReady to summarize."}}},
            model=model.id,
            provider=model.provider,
            api=model.api,
        ),
    ]


def _validate_openai_like_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = DefaultMessageCodec().encode_messages(messages, StreamOptions())
    normalized = normalize_openai_compatible_messages(model, payloads)
    testcase.assertGreaterEqual(len(normalized), 1)
    testcase.assertTrue(all("role" in payload for payload in normalized))


def _validate_minimax_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = MiniMaxMessageCodec().encode_messages(messages, StreamOptions())
    normalized = normalize_openai_compatible_messages(model, payloads)
    testcase.assertGreaterEqual(len(normalized), 1)
    assistant_payloads = [payload for payload in normalized if payload.get("role") == "assistant"]
    if assistant_payloads:
        testcase.assertTrue(all("content" in payload for payload in assistant_payloads))
    else:
        testcase.assertTrue(all(payload.get("role") != "assistant" for payload in normalized))


def _validate_ark_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = ArkMessageCodec().encode_messages(messages, StreamOptions())
    normalized = normalize_openai_compatible_messages(model, payloads)
    testcase.assertGreaterEqual(len(normalized), 1)
    for payload in normalized:
        for tool_call in payload.get("tool_calls", []):
            testcase.assertIsInstance(tool_call["function"]["arguments"], str)


def _validate_anthropic_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = AnthropicMessageCodec().encode_messages(messages, StreamOptions())
    testcase.assertGreaterEqual(len(payloads), 1)
    testcase.assertTrue(all("role" in payload for payload in payloads))
    assistant_payloads = [payload for payload in payloads if payload.get("role") == "assistant" and isinstance(payload.get("content"), list)]
    testcase.assertTrue(assistant_payloads or any(payload.get("role") == "user" for payload in payloads))


def _validate_google_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = GoogleMessageCodec().encode_messages(messages, StreamOptions())
    history, last_user = _convert_messages(payloads)
    testcase.assertIsInstance(history, list)
    if last_user is not None:
        testcase.assertIn(last_user[0], {"user", "model"})
        testcase.assertIsInstance(last_user[1], list)


def _validate_azure_case(testcase: unittest.TestCase, model: Model, messages: List[Message]) -> None:
    payloads = AzureMessageCodec().encode_messages(messages, StreamOptions())
    azure_input = _messages_to_azure_input(payloads, allow_system=False)
    testcase.assertIsInstance(azure_input, list)
    testcase.assertGreaterEqual(len(azure_input), 1)
    testcase.assertTrue(all(("role" in item) or (item.get("type") == "function_call_output") for item in azure_input))


class CrossProviderHandoffMatrixTests(unittest.TestCase):
    def test_cross_provider_handoff_matrix(self):
        source_factories: Dict[str, SourceFactory] = {
            "openai_tool_history": _source_openai_tool_history,
            "google_provider_state": _source_google_provider_state,
            "azure_continuation": _source_azure_continuation,
            "minimax_reasoning": _source_minimax_reasoning,
        }
        target_validators: Dict[str, Tuple[Model, TargetValidator]] = {
            "openai": (Model(id="gpt-5.4", provider="openai", api="openai-chat"), _validate_openai_like_case),
            "azure": (Model(id="gpt-5.4", provider="azure", api="azure-responses"), _validate_azure_case),
            "anthropic": (Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages"), _validate_anthropic_case),
            "google": (Model(id="gemini-3.1-pro-preview", provider="google", api="google-gemini"), _validate_google_case),
            "minimax": (Model(id="minimax", provider="minimax", api="minimax-local"), _validate_minimax_case),
            "ark": (Model(id="doubao-seed-2-0-lite-260215", provider="volcengine", api="ark"), _validate_ark_case),
        }

        for source_name, source_factory in source_factories.items():
            source_messages = source_factory()
            for target_name, (target_model, validator) in target_validators.items():
                with self.subTest(source=source_name, target=target_name):
                    result = handoff_messages(source_messages, target_model)
                    self.assertGreaterEqual(len(result.messages), 1)
                    validator(self, target_model, result.messages)
                    self.assertIsInstance(result.diagnostics, list)


if __name__ == "__main__":
    unittest.main()
