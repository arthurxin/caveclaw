import asyncio
import unittest
from typing import Any, Dict, List, Optional

from agent_core.assistant_messages import (
    AgentContext,
    AgentMessage,
    AgentTool,
    AgentToolUpdate,
    Message,
    RuntimeDeltaOp,
    RuntimeSnapshotBlock,
    RuntimeVariable,
    TextBlock,
    ToolResult,
)
from agent_core.core import run_loop
from agent_core.llm_provider import Model, api_provider_registry


class RecordingTool(AgentTool):
    def __init__(self):
        super().__init__("record_value", "Record a value", {"value": {"type": "string"}}, "Record")
        self.seen_message_roles: List[str] = []

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        context: AgentContext,
        on_update=None,
    ) -> ToolResult:
        self.seen_message_roles = [message.role for message in context.messages if hasattr(message, "role")]
        return ToolResult(
            content=f"stored {params['value']}",
            state_delta={"last_value": params["value"]},
        )


class UpdatingTool(AgentTool):
    def __init__(self):
        super().__init__("stream_value", "Stream a value", {"value": {"type": "string"}}, "Stream")

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        context: AgentContext,
        on_update=None,
        signal=None,
    ) -> ToolResult:
        if on_update:
            on_update(AgentToolUpdate(content="starting"))
            await asyncio.sleep(0)
            on_update(AgentToolUpdate(content="halfway"))
        return ToolResult(
            content="finished",
            state_delta={"streamed_value": params["value"]},
        )


class StagedRuntimeTool(AgentTool):
    def __init__(self):
        super().__init__(
            "stage_runtime",
            "Stage runtime ops",
            {"value": {"type": "string"}},
            "Stage",
            reads=["seed"],
            writes=["final_value"],
            temp_outputs=["draft_value"],
        )
        self.runtime_visible_during_execute: Optional[bool] = None
        self.selection_seen_during_execute = None

    async def execute(
        self,
        tool_call_id: str,
        params: Dict[str, Any],
        context: AgentContext,
        on_update=None,
        signal=None,
    ) -> ToolResult:
        if on_update:
            on_update(
                AgentToolUpdate(
                    content="staging",
                    runtime_ops=[RuntimeDeltaOp(op="set", key="draft_value", value=params["value"])],
                )
            )
        await asyncio.sleep(0)
        self.runtime_visible_during_execute = "draft_value" in context.runtime.variables
        self.selection_seen_during_execute = context.active_tool_selection.to_payload() if context.active_tool_selection else None
        return ToolResult(
            content="staged",
            runtime_ops=[RuntimeDeltaOp(op="set", key="final_value", value=params["value"])],
        )


class FakeProvider:
    api = "fake-provider"

    def __init__(self):
        self.calls = 0
        self.last_tools: Optional[List[AgentTool]] = None

    async def stream(self, model, messages, options, api_key=None):
        self.calls += 1
        self.last_tools = options.tools
        if self.calls == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "record_value",
                        "arguments": {"value": "alpha"},
                    }
                ]
            }
            return
        yield {"content": "done"}


class ProviderConfig:
    max_rounds = 3
    max_consecutive_tool_failures = 3

    def __init__(self, model):
        self.model = model
        self.thinking_level = "off"
        self.system_prompt = None

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        return messages

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        return messages

    async def get_steering_messages(self) -> List[AgentMessage]:
        return []

    async def get_followup_messages(self) -> List[AgentMessage]:
        return []

    async def handle_consolidation(self, messages: List[AgentMessage], context: AgentContext):
        return None


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_loop_passes_tools_to_provider_and_executes_them(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model", provider="fake", api=provider.api)
        config = ProviderConfig(model)
        tool = RecordingTool()
        events = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="record something")],
            tools=[tool],
            config=config,
        ):
            events.append(event)

        event_types = [event.type for event in events]
        self.assertIn("tool_execution_success", event_types)
        self.assertIn("agent_end", event_types)
        self.assertIsNotNone(provider.last_tools)
        self.assertEqual(provider.last_tools[0].name, "record_value")
        self.assertIn("assistant", tool.seen_message_roles)
        self.assertIn("user", tool.seen_message_roles)

    async def test_run_loop_emits_message_update_events(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model-updates", provider="fake", api=provider.api)
        config = ProviderConfig(model)
        event_types = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="record something")],
            tools=[RecordingTool()],
            config=config,
        ):
            event_types.append(event.type)

        self.assertIn("message_start", event_types)
        self.assertIn("message_update", event_types)
        self.assertIn("message_end", event_types)

    async def test_run_loop_emits_tool_execution_update_events(self):
        async def tool_stream(_messages):
            yield {
                "tool_calls": [
                    {
                        "id": "call_stream",
                        "name": "stream_value",
                        "arguments": {"value": "beta"},
                    }
                ]
            }
            yield {"content": "done"}

        config = ProviderConfig(model=None)
        event_types = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="stream something")],
            tools=[UpdatingTool()],
            config=config,
            stream_fn=tool_stream,
        ):
            event_types.append(event.type)

        self.assertIn("tool_execution_update", event_types)
        self.assertIn("tool_execution_success", event_types)

    async def test_run_loop_preserves_raw_content_for_provider_replay(self):
        async def raw_content_stream(_messages):
            yield {
                "raw_content": "<think>plan</think>\nfinal answer",
                "reasoning": "plan",
                "content": "final answer",
            }

        config = ProviderConfig(model=None)
        final_messages = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="show raw content")],
            tools=[],
            config=config,
            stream_fn=raw_content_stream,
        ):
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        assistant_messages = [message for message in final_messages if getattr(message, "role", None) == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0].raw_content, "<think>plan</think>\nfinal answer")
        self.assertEqual(assistant_messages[0].content, "final answer\nplan")
        self.assertEqual(assistant_messages[0].to_dict()["content"], "<think>plan</think>\nfinal answer")

    async def test_run_loop_can_continue_after_consolidation_hook(self):
        class ConsolidatingConfig(ProviderConfig):
            max_rounds = 1

            def __init__(self, model=None):
                super().__init__(model)
                self.did_compact = False

            async def handle_consolidation(self, messages, context):
                if self.did_compact:
                    return None
                self.did_compact = True
                return [Message(role="user", content="compacted context")]

        async def looping_stream(messages):
            user_messages = [message for message in messages if getattr(message, "role", None) == "user"]
            if user_messages and user_messages[-1].content == "compacted context":
                yield {"content": "done after compaction"}
                return
            yield {"tool_calls": [{"id": "call_loop", "name": "record_value", "arguments": {"value": "gamma"}}]}

        config = ConsolidatingConfig()
        event_types = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="loop forever")],
            tools=[RecordingTool()],
            config=config,
            stream_fn=looping_stream,
        ):
            event_types.append(event.type)

        self.assertIn("consolidation_applied", event_types)
        self.assertIn("agent_end", event_types)

    async def test_run_loop_stages_runtime_ops_until_batch_commit(self):
        async def staged_stream(messages):
            if any(getattr(message, "role", None) == "tool" for message in messages):
                yield {"content": "done"}
                return
            yield {"tool_calls": [{"id": "call_stage", "name": "stage_runtime", "arguments": {"value": "delta"}}]}

        config = ProviderConfig(model=None)
        tool = StagedRuntimeTool()
        initial_runtime = None
        final_messages = []
        tool_start_payload = None

        from agent_core.assistant_messages import RuntimeState

        initial_runtime = RuntimeState()
        initial_runtime.set_variable("seed", "existing", llm_view=[TextBlock(text="seed variable")])

        async for event in run_loop(
            context_messages=[Message(role="user", content="stage runtime")],
            tools=[tool],
            config=config,
            stream_fn=staged_stream,
            runtime=initial_runtime,
        ):
            if event.type == "tool_execution_start":
                tool_start_payload = event.data
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        self.assertFalse(tool.runtime_visible_during_execute)
        self.assertIsNotNone(tool.selection_seen_during_execute)
        self.assertEqual(tool.selection_seen_during_execute["reads"], ["seed"])
        self.assertEqual(tool.selection_seen_during_execute["writes"], ["final_value"])
        self.assertEqual(tool.selection_seen_during_execute["temp_outputs"], ["draft_value"])
        self.assertEqual(tool.selection_seen_during_execute["input_variables"]["seed"]["kind"], "opaque")
        self.assertIsNotNone(tool_start_payload)
        self.assertEqual(tool_start_payload["temp_outputs"], ["draft_value"])
        self.assertEqual(tool_start_payload["runtime_selection"]["reads"], ["seed"])
        self.assertEqual(tool_start_payload["runtime_selection"]["missing_reads"], [])

        runtime_messages = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("runtime_injected")
        ]
        self.assertEqual(len(runtime_messages), 1)
        runtime_block = runtime_messages[0].content_blocks[0]
        self.assertIsInstance(runtime_block, RuntimeSnapshotBlock)
        runtime_keys = [entry.key for entry in runtime_block.entries]
        self.assertEqual(sorted(runtime_keys), ["draft_value", "final_value", "seed"])

        worklogs = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("worklog")
        ]
        self.assertEqual(len(worklogs), 1)
        self.assertIn("draft_value", worklogs[0].content)
        self.assertIn("final_value", worklogs[0].content)


if __name__ == "__main__":
    unittest.main()
