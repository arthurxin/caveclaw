import unittest

from agent_core import (
    AgentSessionContext,
    HandoffOptions,
    Message,
    ResolvedHandoffOptions,
    TransformMessagesOptions,
    RuntimeState,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResultMessage,
    handoff_messages,
    handoff_session_context,
    resolve_handoff_options,
)
from agent_core.assistant_messages import AssistantMessage, RuntimeSnapshotBlock, RuntimeSnapshotEntry, transform_messages_with_result
from agent_core.core import AgentHostContext, IPythonProgramRuntime
from agent_core.llm_provider import Model, ModelCompat


class HandoffTests(unittest.TestCase):
    def test_handoff_messages_rewinds_unsafe_trailing_tool_trajectory_by_default(self):
        source_model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="private plan"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="dataset",
                            version=4,
                            summary_blocks=[TextBlock(text="ready for analysis")],
                        )
                    ]
                ),
                ToolCallBlock(id="call|handoff", name="lookup", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|handoff", name="lookup", arguments={"city": "Beijing"})],
            provider_state={"openai": {"response_id": "resp_1"}},
            raw_content="<think>private plan</think>\nlookup",
            stop_reason="tool_use",
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
        )
        tool_result = ToolResultMessage(tool_call_id="call|handoff", name="lookup", content="sunny")

        result = handoff_messages([Message(role="user", content="hello"), assistant, tool_result], target_model)

        self.assertEqual(result.target_model, target_model)
        self.assertEqual(result.resolved_options.rewind_policy, "previous-round-end")
        self.assertTrue(result.rewind_applied)
        self.assertEqual(result.original_message_count, 3)
        self.assertEqual(result.rewound_message_count, 1)
        self.assertEqual(result.rewind_from_index, 1)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].content, "hello")
        diagnostic_codes = {diagnostic.code for diagnostic in result.diagnostics}
        self.assertIn("rewind_applied", diagnostic_codes)
        self.assertTrue(any("rewound transcript" in warning for warning in result.warnings))

    def test_handoff_messages_can_disable_rewind_and_transform_cross_provider_history(self):
        source_model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="private plan"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="dataset",
                            version=4,
                            summary_blocks=[TextBlock(text="ready for analysis")],
                        )
                    ]
                ),
                ToolCallBlock(id="call|handoff", name="lookup", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|handoff", name="lookup", arguments={"city": "Beijing"})],
            provider_state={"openai": {"response_id": "resp_1"}},
            raw_content="<think>private plan</think>\nlookup",
            stop_reason="tool_use",
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
        )
        tool_result = ToolResultMessage(tool_call_id="call|handoff", name="lookup", content="sunny")

        result = handoff_messages(
            [Message(role="user", content="hello"), assistant, tool_result],
            target_model,
            options=HandoffOptions(rewind_policy="none"),
        )

        self.assertEqual(result.resolved_options.provider_state_policy, "same-model-only")
        self.assertEqual(result.resolved_options.runtime_snapshot_policy, "same-model-only")
        self.assertFalse(result.rewind_applied)
        self.assertEqual(result.provider_state_drops, 1)
        self.assertEqual(result.runtime_snapshot_conversions, 1)
        self.assertEqual(result.synthetic_tool_results_added, 0)
        self.assertEqual(result.tool_call_id_rewrites[0].message_index, 1)
        self.assertEqual(result.tool_call_id_rewrites[0].tool_name, "lookup")
        self.assertEqual(result.provider_state_drop_details[0].message_index, 1)
        self.assertEqual(result.provider_state_drop_details[0].namespaces, ["openai"])
        self.assertEqual(result.runtime_block_conversion_details[0].block_type, "runtime_snapshot")
        diagnostic_codes = {diagnostic.code for diagnostic in result.diagnostics}
        self.assertIn("provider_state_dropped", diagnostic_codes)
        self.assertIn("runtime_block_converted", diagnostic_codes)
        self.assertIn("tool_call_id_rewritten", diagnostic_codes)
        self.assertTrue(any("dropped provider_state" in warning for warning in result.warnings))
        transformed_assistant = result.messages[1]
        transformed_tool_result = result.messages[2]
        self.assertEqual(transformed_assistant.provider_state, None)
        self.assertEqual(transformed_assistant.raw_content, None)
        self.assertEqual(transformed_assistant.content_blocks[0].text, "private plan")
        self.assertIn("Runtime Snapshot:", transformed_assistant.content_blocks[1].text)
        self.assertNotEqual(transformed_assistant.tool_calls[0].id, "call|handoff")
        self.assertEqual(transformed_tool_result.tool_call_id, transformed_assistant.tool_calls[0].id)

    def test_handoff_messages_can_preserve_same_model_provider_state(self):
        model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        assistant = AssistantMessage(
            content_blocks=[ThinkingBlock(thinking="keep replay", signature="sig-1")],
            provider_state={"openai": {"response_id": "resp_1"}},
            model=model.id,
            provider=model.provider,
            api=model.api,
        )

        result = handoff_messages(
            [Message(role="user", content="hello"), assistant],
            model,
            options=HandoffOptions(provider_state_policy="preserve"),
        )

        transformed_assistant = result.messages[1]
        self.assertEqual(transformed_assistant.provider_state["openai"]["response_id"], "resp_1")
        self.assertIsInstance(transformed_assistant.content_blocks[0], ThinkingBlock)

    def test_handoff_messages_can_disable_synthetic_tool_results_and_drop_runtime_snapshot(self):
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        assistant = AssistantMessage(
            content_blocks=[
                RuntimeSnapshotBlock(
                    entries=[RuntimeSnapshotEntry(key="dataset", version=1, summary_blocks=[TextBlock(text="ready")])]
                ),
                ToolCallBlock(id="call|missing", name="lookup", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|missing", name="lookup", arguments={"city": "Beijing"})],
            stop_reason="tool_use",
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )

        result = handoff_messages(
            [Message(role="user", content="hello"), assistant, Message(role="user", content="next")],
            target_model,
            options=HandoffOptions(
                rewind_policy="none",
                runtime_snapshot_policy="drop",
                synthetic_tool_result_policy="skip",
            ),
        )

        transformed_assistant = result.messages[1]
        self.assertEqual(result.synthetic_tool_results_added, 0)
        self.assertEqual(result.runtime_snapshot_conversions, 0)
        self.assertTrue(all(not isinstance(block, RuntimeSnapshotBlock) for block in transformed_assistant.content_blocks))
        self.assertEqual(len(result.messages), 3)

    def test_handoff_session_context_keeps_host_resources_and_metadata(self):
        source_model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        runtime = RuntimeState()
        python_runtime = IPythonProgramRuntime(initial_namespace={"seed": "python"})
        host = AgentHostContext(runtime=runtime, python_runtime=python_runtime, api_key="session-key")
        assistant = AssistantMessage(
            content_blocks=[ThinkingBlock(thinking="plan")],
            model=source_model.id,
            provider=source_model.provider,
            api=source_model.api,
        )
        session_context = AgentSessionContext(
            messages=[Message(role="user", content="start"), assistant],
            tools=[],
            host=host,
            metadata={"trace_id": "abc"},
        )

        result = handoff_session_context(session_context, target_model)

        self.assertIsNotNone(result.session_context)
        self.assertIs(result.session_context.host, host)
        self.assertIs(result.session_context.host.runtime, runtime)
        self.assertIs(result.session_context.host.python_runtime, python_runtime)
        self.assertEqual(result.session_context.metadata["trace_id"], "abc")
        transformed_assistant = result.session_context.messages[1]
        self.assertEqual(transformed_assistant.content_blocks[0].text, "plan")

    def test_transform_messages_with_result_reports_policy_effects(self):
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="plan"),
                RuntimeSnapshotBlock(
                    entries=[RuntimeSnapshotEntry(key="dataset", version=2, summary_blocks=[TextBlock(text="ready")])]
                ),
                ToolCallBlock(id="call|1", name="lookup", arguments={"city": "Beijing"}),
            ],
            tool_calls=[ToolCall(id="call|1", name="lookup", arguments={"city": "Beijing"})],
            provider_state={"openai": {"response_id": "resp_1"}},
            stop_reason="tool_use",
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )
        aborted = AssistantMessage(
            content_blocks=[TextBlock(text="partial")],
            stop_reason="aborted",
            model="gpt-5.4",
            provider="openai",
            api="openai-chat",
        )

        result = transform_messages_with_result(
            [aborted, assistant, Message(role="user", content="next")],
            target_model,
            options=TransformMessagesOptions(),
        )

        self.assertEqual(result.skipped_assistant_messages, 1)
        self.assertEqual(result.provider_state_drops, 1)
        self.assertEqual(result.runtime_snapshot_conversions, 1)
        self.assertEqual(result.synthetic_tool_results_added, 1)
        self.assertIn("call|1", result.tool_call_id_map)
        self.assertEqual(result.skipped_assistant_details[0].stop_reason, "aborted")
        self.assertEqual(result.provider_state_drop_details[0].namespaces, ["openai"])
        self.assertEqual(result.synthetic_tool_result_details[0].tool_name, "lookup")
        self.assertEqual(result.runtime_block_conversion_details[0].block_type, "runtime_snapshot")

    def test_resolve_handoff_options_uses_target_model_compat_defaults(self):
        target_model = Model(
            id="text-thinking-model",
            provider="strict-provider",
            api="strict-api",
            compat=ModelCompat(requiresThinkingAsText=True),
        )

        resolved = resolve_handoff_options(target_model)

        self.assertIsInstance(resolved, ResolvedHandoffOptions)
        self.assertEqual(resolved.rewind_policy, "previous-round-end")
        self.assertEqual(resolved.provider_state_policy, "drop")
        self.assertEqual(resolved.runtime_snapshot_policy, "render-text")
        self.assertEqual(resolved.synthetic_tool_result_policy, "insert")
        self.assertEqual(resolved.assistant_failure_policy, "skip")

    def test_handoff_messages_uses_compat_defaults_to_degrade_same_model_structured_state(self):
        target_model = Model(
            id="text-thinking-model",
            provider="strict-provider",
            api="strict-api",
            compat=ModelCompat(requiresThinkingAsText=True),
        )
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="private plan", signature="sig-1"),
                RuntimeSnapshotBlock(
                    entries=[RuntimeSnapshotEntry(key="dataset", version=1, summary_blocks=[TextBlock(text="ready")])]
                ),
            ],
            provider_state={"strict-provider": {"replay": "opaque"}},
            raw_content="<think>private plan</think>\nready",
            model=target_model.id,
            provider=target_model.provider,
            api=target_model.api,
        )

        result = handoff_messages([Message(role="user", content="hello"), assistant], target_model)

        transformed_assistant = result.messages[1]
        self.assertEqual(result.resolved_options.provider_state_policy, "drop")
        self.assertEqual(result.resolved_options.runtime_snapshot_policy, "render-text")
        self.assertEqual(transformed_assistant.provider_state, None)
        self.assertEqual(transformed_assistant.raw_content, None)
        self.assertTrue(all(not isinstance(block, ThinkingBlock) for block in transformed_assistant.content_blocks))
        self.assertEqual(transformed_assistant.content_blocks[0].text, "private plan")
        self.assertIn("Runtime Snapshot:", transformed_assistant.content_blocks[1].text)
        self.assertEqual(result.provider_state_drop_details[0].namespaces, ["strict-provider"])
        self.assertEqual(result.runtime_block_conversion_details[0].mode, "render-text")
        self.assertTrue(any(diagnostic.code == "provider_state_dropped" for diagnostic in result.diagnostics))

    def test_handoff_keeps_latest_safe_user_turn_but_rewinds_unsafe_assistant_suffix(self):
        target_model = Model(id="gpt-5.4", provider="azure", api="azure-responses")
        messages = [
            Message(role="user", content="round one"),
            AssistantMessage(content_blocks=[TextBlock(text="done")], model="gpt-5.4", provider="openai", api="openai-chat"),
            Message(role="user", content="round two"),
            AssistantMessage(
                content_blocks=[ToolCallBlock(id="call|2", name="lookup", arguments={"topic": "x"})],
                tool_calls=[ToolCall(id="call|2", name="lookup", arguments={"topic": "x"})],
                stop_reason="tool_use",
                model="gpt-5.4",
                provider="openai",
                api="openai-chat",
            ),
        ]

        result = handoff_messages(messages, target_model)

        self.assertTrue(result.rewind_applied)
        self.assertEqual([message.content for message in result.messages if isinstance(message, Message)], ["round one", "done", "round two"])
        self.assertEqual(result.rewound_message_count, 3)
        self.assertEqual(result.rewind_from_index, 3)


if __name__ == "__main__":
    unittest.main()
