import unittest

from agent_core.assistant_messages import Message, ThinkingBlock, stream_assistant_response
from agent_core.llm_provider.providers.minimax_provider import MiniMaxProvider


class DummyConfig:
    model = None
    thinking_level = "off"
    system_prompt = None


class MiniMaxProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_assistant_response_keeps_single_thinking_block_for_minimax_pattern(self):
        async def minimax_style_stream(_messages):
            yield {"reasoning": "plan"}
            yield {"raw_content": "<think>plan</think>\nfinal answer", "content": "final answer"}

        assistant_message = await stream_assistant_response(
            [Message(role="user", content="hello")],
            tools=[],
            config=DummyConfig(),
            stream_fn=minimax_style_stream,
        )

        thinking_blocks = [block for block in assistant_message.content_blocks if isinstance(block, ThinkingBlock)]
        self.assertEqual(len(thinking_blocks), 1)
        self.assertEqual(thinking_blocks[0].thinking, "plan")
        self.assertEqual(assistant_message.raw_content, "<think>plan</think>\nfinal answer")
        self.assertEqual(assistant_message.to_dict()["content"], "<think>plan</think>\nfinal answer")

    def test_extract_reasoning_and_content_preserves_raw_content(self):
        parsed = MiniMaxProvider._extract_reasoning_and_content("<think>consider</think>\nhello")

        self.assertEqual(parsed["reasoning"], "consider")
        self.assertEqual(parsed["content"], "hello")
        self.assertEqual(parsed["raw_content"], "<think>consider</think>\nhello")


if __name__ == "__main__":
    unittest.main()
