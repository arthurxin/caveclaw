import asyncio
import unittest
from typing import Any, Dict, List, Optional

from agent_core.assistant_messages import (
    AgentContext,
    AssistantDelta,
    AgentMessage,
    AgentTool,
    AgentToolUpdate,
    Message,
    RuntimeDeltaOp,
    RuntimeSnapshotBlock,
    RuntimeState,
    RuntimeVariable,
    TextBlock,
    ToolResult,
)
from agent_core.core import DefaultPythonProgramExecutionController, PythonRuntimeBridgeResult, run_loop
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
        self.last_api_key = None

    async def stream(self, model, messages, options, api_key=None):
        self.calls += 1
        self.last_tools = options.tools
        self.last_api_key = api_key
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
    async def test_run_loop_passes_python_runtime_into_python_lane_namespace(self):
        class MarkerRuntime:
            def __init__(self, label):
                self.label = label

        async def python_stream(messages):
            if any(
                getattr(message, "metadata", {}).get("python_program_execution")
                for message in messages
                if hasattr(message, "metadata")
            ):
                yield {"content": "python lane complete"}
                return
            yield {"content": '```python\nprint(python_runtime.label)\n```'}

        config = ProviderConfig(model=None)
        config.python_program_execution = True
        config.python_program_backend = "python"
        marker_runtime = MarkerRuntime("shared-python-runtime")
        final_messages = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="show python runtime")],
            tools=[],
            config=config,
            stream_fn=python_stream,
            python_runtime=marker_runtime,
        ):
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        python_messages = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("python_program_execution")
        ]
        self.assertEqual(len(python_messages), 1)
        self.assertIn("shared-python-runtime", python_messages[0].content)

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

    async def test_run_loop_uses_config_get_api_key_before_provider_stream(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-model-api-key", provider="fake", api=provider.api)
        config = ProviderConfig(model)

        async def get_api_key(provider_name):
            return f"{provider_name}-loop-key"

        config.get_api_key = get_api_key

        async for _event in run_loop(
            context_messages=[Message(role="user", content="hello")],
            tools=[],
            config=config,
        ):
            pass

        self.assertEqual(provider.last_api_key, "fake-loop-key")

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

    async def test_run_loop_exposes_event_accessors_for_application_consumers(self):
        async def tool_stream(_messages):
            yield {
                "tool_calls": [
                    {
                        "id": "call_accessors",
                        "name": "stream_value",
                        "arguments": {"value": "gamma"},
                    }
                ]
            }
            yield {"content": "done"}

        config = ProviderConfig(model=None)
        events = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="exercise accessors")],
            tools=[UpdatingTool()],
            config=config,
            stream_fn=tool_stream,
        ):
            events.append(event)

        turn_start = next(event for event in events if event.type == "turn_start")
        self.assertEqual(turn_start.round, 1)

        message_update = next(event for event in events if event.type == "message_update")
        self.assertIsNotNone(message_update.message)
        self.assertIsInstance(message_update.delta, AssistantDelta)
        self.assertIs(message_update.assistant_delta, message_update.delta)
        self.assertTrue(message_update.delta.has_tool_calls)

        tool_start = next(event for event in events if event.type == "tool_execution_start")
        self.assertEqual(tool_start.tool_call_id, "call_accessors")
        self.assertEqual(tool_start.tool_name, "stream_value")
        self.assertEqual(tool_start.args, {"value": "gamma"})

        tool_update = next(event for event in events if event.type == "tool_execution_update")
        self.assertEqual(tool_update.tool_call_id, "call_accessors")
        self.assertEqual(tool_update.tool_name, "stream_value")
        self.assertIsNotNone(tool_update.partial_result)

        tool_success = next(event for event in events if event.type == "tool_execution_success")
        self.assertEqual(tool_success.tool_call_id, "call_accessors")
        self.assertEqual(tool_success.tool_name, "stream_value")
        self.assertEqual(tool_success.result, "finished")
        self.assertFalse(tool_success.is_error)

        turn_end = next(event for event in events if event.type == "turn_end" and event.tool_results)
        self.assertEqual(len(turn_end.tool_results), 1)

        agent_end = next(event for event in events if event.type == "agent_end")
        self.assertTrue(agent_end.messages)

    async def test_run_loop_executes_python_program_lane_separately(self):
        async def python_stream(messages):
            if any(
                getattr(message, "metadata", {}).get("python_program_execution")
                for message in messages
                if hasattr(message, "metadata")
            ):
                yield {"content": "python lane complete"}
                return
            yield {
                "content": 'I will compute it.\n```python\nprint("hello from python lane")\nvalue = 7\n```',
            }

        config = ProviderConfig(model=None)
        config.python_program_execution = True
        config.python_program_backend = "python"
        event_types = []
        final_messages = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="run python")],
            tools=[],
            config=config,
            stream_fn=python_stream,
        ):
            event_types.append(event.type)
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        self.assertIn("python_program_execution_start", event_types)
        self.assertIn("python_program_execution_success", event_types)
        self.assertNotIn("tool_execution_start", event_types)

        python_messages = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("python_program_execution")
        ]
        self.assertEqual(len(python_messages), 1)
        self.assertIn("hello from python lane", python_messages[0].content)

        worklogs = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("lane") == "python_program"
        ]
        self.assertEqual(len(worklogs), 0)

    async def test_run_loop_keeps_native_tool_calling_separate_from_python_lane(self):
        async def mixed_stream(messages):
            if any(getattr(message, "role", None) == "tool" for message in messages):
                yield {"content": "native tool lane complete"}
                return
            yield {
                "content": '```python\nprint("should not run")\n```',
                "tool_calls": [
                    {
                        "id": "call_native",
                        "name": "record_value",
                        "arguments": {"value": "native"},
                    }
                ],
            }

        config = ProviderConfig(model=None)
        event_types = []
        final_messages = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="prefer native tool")],
            tools=[RecordingTool()],
            config=config,
            stream_fn=mixed_stream,
        ):
            event_types.append(event.type)
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        self.assertIn("tool_execution_start", event_types)
        self.assertIn("tool_execution_success", event_types)
        self.assertNotIn("python_program_execution_start", event_types)

        python_messages = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("python_program_execution")
        ]
        self.assertEqual(len(python_messages), 0)

    async def test_run_loop_can_disable_python_program_lane_with_switch(self):
        async def python_stream(_messages):
            yield {"content": '```python\nprint("disabled")\n```'}

        config = ProviderConfig(model=None)
        config.python_program_execution = False
        event_types = []
        final_messages = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="do not run python lane")],
            tools=[],
            config=config,
            stream_fn=python_stream,
        ):
            event_types.append(event.type)
            if event.type == "agent_end":
                final_messages = event.data["messages"]

        self.assertNotIn("python_program_execution_start", event_types)
        python_messages = [
            message
            for message in final_messages
            if isinstance(message, Message) and message.metadata.get("python_program_execution")
        ]
        self.assertEqual(len(python_messages), 0)

    async def test_run_loop_can_use_dedicated_python_program_controller(self):
        class PreferPythonController(DefaultPythonProgramExecutionController):
            def __init__(self):
                super().__init__(enabled=True, require_no_tool_calls=False)

        async def mixed_stream(messages):
            if any(
                getattr(message, "metadata", {}).get("python_program_execution")
                for message in messages
                if hasattr(message, "metadata")
            ):
                yield {"content": "python controller complete"}
                return
            yield {
                "content": '```python\nprint("controller ran")\n```',
                "tool_calls": [
                    {
                        "id": "call_native_ignored",
                        "name": "record_value",
                        "arguments": {"value": "ignored"},
                    }
                ],
            }

        config = ProviderConfig(model=None)
        config.python_program_backend = "python"
        config.python_program_execution = PreferPythonController()
        event_types = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="prefer controller lane")],
            tools=[RecordingTool()],
            config=config,
            stream_fn=mixed_stream,
        ):
            event_types.append(event.type)

        self.assertIn("python_program_execution_start", event_types)
        self.assertIn("python_program_execution_success", event_types)
        self.assertNotIn("tool_execution_start", event_types)

    async def test_run_loop_bridges_python_namespace_into_runtime(self):
        async def python_stream(messages):
            if any(
                isinstance(message, Message) and message.metadata.get("runtime_injected")
                for message in messages
            ):
                yield {"content": "runtime bridge observed"}
                return
            yield {
                "content": (
                    "```python\n"
                    "numbers = [1, 2, 3]\n"
                    "summary = {'total': sum(numbers), 'count': len(numbers)}\n"
                    "print(summary['total'])\n"
                    "```"
                ),
            }

        config = ProviderConfig(model=None)
        config.python_program_execution = True
        config.python_program_backend = "python"
        runtime = RuntimeState()
        synced_variables = None

        async for event in run_loop(
            context_messages=[Message(role="user", content="analyze numbers")],
            tools=[],
            config=config,
            stream_fn=python_stream,
            runtime=runtime,
        ):
            if event.type == "python_program_execution_success":
                synced_variables = event.get("synced_variables")

        self.assertEqual(sorted(runtime.variables.keys()), ["numbers", "summary"])
        self.assertEqual(runtime.variables["summary"].raw_value["total"], 6)
        self.assertEqual(synced_variables, ["numbers", "summary"])

    async def test_run_loop_can_use_custom_python_runtime_bridge(self):
        class RenameBridge:
            async def build_runtime_bridge_result(
                self,
                *,
                execution_request,
                execution_result,
                agent_context,
                python_block,
            ):
                del execution_request
                del agent_context
                del python_block
                return PythonRuntimeBridgeResult(
                    runtime_ops=[
                        RuntimeDeltaOp(
                            op="set",
                            key="analysis.answer",
                            value=execution_result.namespace["answer"],
                            metadata={"python_bridge": True},
                        )
                    ],
                    synced_variables=["analysis.answer"],
                )

        async def python_stream(_messages):
            yield {"content": '```python\nanswer = 42\n```'}

        config = ProviderConfig(model=None)
        config.python_program_execution = True
        config.python_program_backend = "python"
        config.python_runtime_bridge = RenameBridge()
        runtime = RuntimeState()

        async for _event in run_loop(
            context_messages=[Message(role="user", content="compute answer")],
            tools=[],
            config=config,
            stream_fn=python_stream,
            runtime=runtime,
        ):
            pass

        self.assertIn("analysis.answer", runtime.variables)
        self.assertEqual(runtime.variables["analysis.answer"].raw_value, 42)

    async def test_run_loop_does_not_call_controller_when_python_lane_is_disabled(self):
        class ExplodingController:
            enabled = False

            def select_python_block(self, assistant_message):
                raise AssertionError("controller should not be called when python lane is disabled")

        async def plain_stream(_messages):
            yield {"content": '```python\nprint("do not inspect by default")\n```'}

        config = ProviderConfig(model=None)
        config.python_program_execution = ExplodingController()
        event_types = []

        async for event in run_loop(
            context_messages=[Message(role="user", content="default should skip parser")],
            tools=[],
            config=config,
            stream_fn=plain_stream,
        ):
            event_types.append(event.type)

        self.assertNotIn("python_program_execution_start", event_types)

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
