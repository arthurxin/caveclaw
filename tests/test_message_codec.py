import unittest

from agent_core.assistant_stream import append_assistant_delta
from agent_core.llm.api_registry import StreamOptions
from agent_core.llm.message_codec import GoogleMessageCodec, MiniMaxMessageCodec
from agent_core.llm.providers.google_provider import _convert_messages
from agent_core.types import AssistantMessage, Message


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

        self.assertEqual(last_user, "next")
        self.assertEqual(history[0]["parts"][0]["text"], "preserved")
        self.assertEqual(history[0]["parts"][0]["thoughtSignature"], "sig-2")

    def test_assistant_delta_merges_provider_state_lists(self):
        message = AssistantMessage(content_blocks=[])

        append_assistant_delta(message, {"provider_state": {"gemini": {"thought_signatures": ["sig-1"]}}})
        append_assistant_delta(message, {"provider_state": {"gemini": {"thought_signatures": ["sig-2"]}}})

        self.assertEqual(message.provider_state["gemini"]["thought_signatures"], ["sig-1", "sig-2"])


if __name__ == "__main__":
    unittest.main()
