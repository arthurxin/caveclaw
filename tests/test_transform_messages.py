import unittest

from agent_core.assistant_messages import (
    AssistantMessage,
    Message,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultMessage,
    stream_assistant_response,
    transform_messages,
)
from agent_core.llm_provider import Model, api_provider_registry


def _build_model(*, provider: str, api: str, model_id: str) -> Model:
    return Model(id=model_id, provider=provider, api=api)


class TransformMessagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_transform_messages_preserves_same_model_provider_state_and_snapshot(self):
        model = _build_model(provider="google", api="google-gemini", model_id="gemini-3.1-pro-preview")
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="plan", signature="sig-1"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="df",
                            version=3,
                            summary_blocks=[TextBlock(text="DataFrame with 3 rows")],
                            metadata={"kind": "table"},
                        )
                    ]
                ),
            ],
            provider_state={"gemini": {"parts": [{"text": "provider replay"}]}},
            model=model.id,
            provider=model.provider,
            api=model.api,
        )

        transformed = transform_messages([assistant], model)

        self.assertEqual(len(transformed), 1)
        transformed_assistant = transformed[0]
        self.assertIsInstance(transformed_assistant.content_blocks[0], ThinkingBlock)
        self.assertIsInstance(transformed_assistant.content_blocks[1], RuntimeSnapshotBlock)
        self.assertEqual(transformed_assistant.provider_state["gemini"]["parts"][0]["text"], "provider replay")

    async def test_transform_messages_degrades_cross_model_thinking_and_runtime_snapshot(self):
        source_model = _build_model(provider="openai", api="openai-chat", model_id="gpt-5.4")
        target_model = _build_model(provider="anthropic", api="anthropic-messages", model_id="claude-4-6-sonnet-20241022")
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="reason through the steps", signature="sig-2"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="table",
                            version=2,
                            summary_blocks=[TextBlock(text="3 columns, 10 rows")],
                            metadata={"kind": "dataframe"},
                        )
                    ]
                ),
            ],
            raw_content="<think>reason through the steps</think>\nanswer",
            provider_state={"openai": {"response_id": "resp_1"}},
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
        )

        transformed = transform_messages([assistant], target_model)

        transformed_assistant = transformed[0]
        self.assertEqual(transformed_assistant.raw_content, None)
        self.assertEqual(transformed_assistant.provider_state, None)
        self.assertTrue(all(not isinstance(block, ThinkingBlock) for block in transformed_assistant.content_blocks))
        self.assertEqual(transformed_assistant.content_blocks[0].text, "reason through the steps")
        self.assertIn("Runtime Snapshot:", transformed_assistant.content_blocks[1].text)
        self.assertIn("table@v2", transformed_assistant.content_blocks[1].text)

    async def test_transform_messages_normalizes_tool_ids_skips_aborted_and_synthesizes_results(self):
        target_model = _build_model(provider="anthropic", api="anthropic-messages", model_id="claude-4-6-sonnet-20241022")
        aborted = AssistantMessage(
            content_blocks=[TextBlock(text="partial")],
            stop_reason="aborted",
            tool_calls=[ToolCall(id="call|aborted", name="broken_tool", arguments={})],
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )
        assistant = AssistantMessage(
            content_blocks=[
                ToolCallBlock(id="call|1/very-long", name="lookup_weather", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|1/very-long", name="lookup_weather", arguments={"city": "Beijing"})],
            stop_reason="tool_use",
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )
        trailing_user = Message(role="user", content="continue")

        transformed = transform_messages([aborted, assistant, trailing_user], target_model)

        self.assertEqual(len(transformed), 3)
        self.assertIsInstance(transformed[0], AssistantMessage)
        self.assertNotEqual(transformed[0].tool_calls[0].id, "call|1/very-long")
        self.assertIsInstance(transformed[1], ToolResultMessage)
        self.assertEqual(transformed[1].tool_call_id, transformed[0].tool_calls[0].id)
        self.assertTrue(transformed[1].is_error)
        self.assertEqual(transformed[1].content, "No result provided")
        self.assertEqual(transformed[2].role, "user")

    async def test_stream_assistant_response_applies_transform_before_codec_encoding(self):
        class CaptureCodec:
            namespace = "capture"

            def __init__(self):
                self.captured_messages = None

            def encode_messages(self, messages, options):
                self.captured_messages = messages
                return [message.to_dict() for message in messages]

            def decode_chunk(self, chunk, assistant_message):
                return dict(chunk)

            def finalize_provider_state(self, assistant_message):
                return None

            def finalize_assistant_message(self, assistant_message):
                return None

        class CaptureProvider:
            api = "capture-transform-provider"

            def __init__(self):
                self.message_codec = CaptureCodec()

            async def stream(self, model, messages, options, api_key=None):
                yield {"content": "done"}

        provider = CaptureProvider()
        api_provider_registry.register(provider)
        target_model = _build_model(provider="anthropic", api=provider.api, model_id="claude-4-6-sonnet-20241022")

        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="private reasoning"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="state",
                            version=1,
                            summary_blocks=[TextBlock(text="ready")],
                        )
                    ]
                ),
                ToolCallBlock(id="call|pipe", name="lookup", arguments={"q": "weather"}),
            ],
            tool_calls=[ToolCall(id="call|pipe", name="lookup", arguments={"q": "weather"})],
            provider_state={"openai": {"response_id": "resp_1"}},
            raw_content="<think>private reasoning</think>\nlookup",
            stop_reason="tool_use",
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )
        tool_result = ToolResultMessage(tool_call_id="call|pipe", name="lookup", content="sunny")

        class Config:
            model = target_model
            system_prompt = None
            thinking_level = "off"

        await stream_assistant_response(
            messages=[Message(role="user", content="hi"), assistant, tool_result],
            tools=[],
            config=Config(),
        )

        captured = provider.message_codec.captured_messages
        self.assertIsNotNone(captured)
        transformed_assistant = captured[1]
        transformed_tool_result = captured[2]
        self.assertEqual(transformed_assistant.provider_state, None)
        self.assertEqual(transformed_assistant.raw_content, None)
        self.assertEqual(transformed_assistant.content_blocks[0].text, "private reasoning")
        self.assertIn("Runtime Snapshot:", transformed_assistant.content_blocks[1].text)
        self.assertNotEqual(transformed_assistant.tool_calls[0].id, "call|pipe")
        self.assertEqual(transformed_tool_result.tool_call_id, transformed_assistant.tool_calls[0].id)


if __name__ == "__main__":
    unittest.main()
