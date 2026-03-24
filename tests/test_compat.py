import unittest

from agent_core.llm_provider import (
    Model,
    ModelCompat,
    build_openai_compatible_tools,
    build_reasoning_payload,
    normalize_openai_compatible_messages,
    requires_assistant_after_tool_result,
    requires_tool_result_name,
    resolve_max_tokens_field,
    supports_developer_role,
    supports_store,
    supports_strict_tool_schema,
    supports_usage_in_streaming,
)
from agent_core.assistant_messages.tools import AgentTool


class CompatTests(unittest.TestCase):
    def test_normalize_openai_messages_inserts_assistant_between_tool_and_user(self):
        model = Model(
            id="test-model",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(requiresAssistantAfterToolResult=True),
        )

        messages = [
            {"role": "assistant", "content": ""},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "user", "content": "continue"},
        ]

        normalized = normalize_openai_compatible_messages(model, messages)

        self.assertEqual(normalized[2]["role"], "assistant")
        self.assertEqual(normalized[2]["content"], "")
        self.assertEqual(normalized[3]["role"], "user")

    def test_build_reasoning_payload_respects_qwen_format(self):
        model = Model(
            id="qwen-test",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(thinkingFormat="qwen"),
        )

        payload = build_reasoning_payload(model, "medium")

        self.assertEqual(payload, {"enable_thinking": True})

    def test_max_tokens_and_streaming_usage_respect_compat(self):
        model = Model(
            id="compat-test",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(maxTokensField="max_completion_tokens", supportsUsageInStreaming=False),
        )

        self.assertEqual(resolve_max_tokens_field(model), "max_completion_tokens")
        self.assertFalse(supports_usage_in_streaming(model))

    def test_developer_role_falls_back_to_system_when_provider_disables_it(self):
        model = Model(
            id="compat-test",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(supportsDeveloperRole=False),
        )

        normalized = normalize_openai_compatible_messages(
            model,
            [{"role": "developer", "content": "Be terse."}],
        )

        self.assertEqual(normalized, [{"role": "system", "content": "Be terse."}])
        self.assertFalse(supports_developer_role(model))

    def test_build_openai_tools_adds_strict_schema_when_enabled(self):
        model = Model(
            id="strict-tool-test",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(supportsStrictToolSchema=True),
        )
        tool = AgentTool(
            name="summarize_csv",
            description="Summarize a CSV file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {"limit": {"type": "integer"}},
                    },
                },
                "required": ["path"],
            },
            label="Summarize CSV",
        )

        payload = build_openai_compatible_tools(model, [tool])

        self.assertTrue(supports_strict_tool_schema(model))
        self.assertEqual(payload[0]["function"]["strict"], True)
        self.assertEqual(payload[0]["function"]["parameters"]["additionalProperties"], False)
        self.assertEqual(
            payload[0]["function"]["parameters"]["properties"]["options"]["additionalProperties"],
            False,
        )

    def test_boolean_compat_helpers_default_and_override(self):
        default_model = Model(id="default", provider="test", api="openai-chat")
        override_model = Model(
            id="override",
            provider="test",
            api="openai-chat",
            compat=ModelCompat(
                supportsStore=True,
                requiresToolResultName=True,
                requiresAssistantAfterToolResult=True,
            ),
        )

        self.assertFalse(supports_store(default_model))
        self.assertTrue(supports_developer_role(default_model))
        self.assertFalse(requires_tool_result_name(default_model))
        self.assertFalse(requires_assistant_after_tool_result(default_model))
        self.assertTrue(supports_store(override_model))
        self.assertTrue(requires_tool_result_name(override_model))
        self.assertTrue(requires_assistant_after_tool_result(override_model))


if __name__ == "__main__":
    unittest.main()
