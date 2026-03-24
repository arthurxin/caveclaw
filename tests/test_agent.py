import unittest

from agent_core import Agent, Message
from agent_core.assistant_messages import (
    AgentContext,
    AgentTool,
    AssistantMessage,
    RuntimeSnapshotBlock,
    RuntimeSnapshotEntry,
    RuntimeState,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolResult,
    ToolResultMessage,
)
from agent_core.core import AgentHostContext, AgentSessionContext, IPythonProgramRuntime
from agent_core.llm_provider import Model, api_provider_registry


class FakeProvider:
    api = "fake-agent-provider"

    def __init__(self):
        self.calls = 0
        self.last_api_key = None
        self.last_options = None

    async def stream(self, model, messages, options, api_key=None):
        self.calls += 1
        self.last_api_key = api_key
        self.last_options = options
        if self.calls == 1:
            yield {"content": "first"}
            return
        yield {"content": "continued"}


class AgentConfig:
    max_rounds = 3
    max_consecutive_tool_failures = 3

    def __init__(self, model):
        self.model = model
        self.thinking_level = "off"
        self.system_prompt = None
        self.temperature = None
        self.max_tokens = None
        self.session_id = None
        self.transport = None
        self.cache_retention = None
        self.headers = None
        self.max_retry_delay_ms = None
        self.metadata = None
        self.thinking_budgets = None
        self.on_payload = None

    async def convert_to_llm(self, messages):
        return messages

    async def transform_context(self, messages):
        return messages

    async def get_steering_messages(self):
        return []

    async def get_followup_messages(self):
        return []


class RuntimeWritingTool(AgentTool):
    def __init__(self):
        super().__init__("write_runtime", "Write runtime", {"value": {"type": "string"}}, "Write")

    async def execute(
        self,
        tool_call_id: str,
        params,
        context: AgentContext,
        on_update=None,
        signal=None,
    ) -> ToolResult:
        return ToolResult(content="stored", state_delta={"agent_value": params["value"]})


class ToolProvider:
    api = "fake-runtime-provider"

    def __init__(self):
        self.calls = 0

    async def stream(self, model, messages, options, api_key=None):
        self.calls += 1
        if any(message.get("role") == "tool" for message in messages):
            yield {"content": "done"}
            return
        yield {"tool_calls": [{"id": "call_runtime", "name": "write_runtime", "arguments": {"value": "omega"}}]}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_prompt_accepts_plain_string_input(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="string-prompt-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        async for _event in agent.prompt("hello from string prompt"):
            pass

        self.assertEqual(agent.state.messages[0].role, "user")
        self.assertEqual(agent.state.messages[0].content, "hello from string prompt")
        self.assertEqual(provider.calls, 1)

    async def test_agent_initializes_runtime_and_python_runtime_by_default(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="default-runtime-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        self.assertIsInstance(agent.state.runtime, RuntimeState)
        self.assertIsInstance(agent.state.python_runtime, IPythonProgramRuntime)

    async def test_agent_continue_run_reuses_existing_context(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-agent-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        first_events = []
        async for event in agent.prompt(Message(role="user", content="hello")):
            first_events.append(event.type)

        self.assertIn("agent_end", first_events)
        self.assertTrue(any(getattr(message, "role", None) == "assistant" for message in agent.state.messages))

        agent.append_message(Message(role="user", content="continue"))
        second_events = []
        async for event in agent.continue_run():
            second_events.append(event.type)

        self.assertIn("agent_end", second_events)
        self.assertEqual(provider.calls, 2)

    async def test_agent_queue_modes_control_delivery(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-queue-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        agent.set_steering_mode("one-at-a-time")
        agent.steer(Message(role="user", content="one"))
        agent.steer(Message(role="user", content="two"))
        first_batch = await agent._get_steering_messages()
        second_batch = await agent._get_steering_messages()

        self.assertEqual(len(first_batch), 1)
        self.assertEqual(len(second_batch), 1)
        self.assertEqual(first_batch[0].content, "one")
        self.assertEqual(second_batch[0].content, "two")

    async def test_agent_queue_helpers_accept_strings_and_batches(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="string-queue-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        agent.set_steering_mode("all")
        agent.steer(["one", Message(role="user", content="two")])
        steering_batch = await agent._get_steering_messages()

        self.assertEqual([message.content for message in steering_batch], ["one", "two"])
        self.assertTrue(all(message.role == "user" for message in steering_batch))

        agent.follow_up(["three", "four"])
        followup_batch = await agent._get_followup_messages()
        self.assertEqual([message.content for message in followup_batch], ["three"])

        agent.set_followup_mode("all")
        agent.follow_up(["five", "six"])
        followup_batch = await agent._get_followup_messages()
        self.assertEqual([message.content for message in followup_batch], ["four", "five", "six"])

    async def test_agent_runtime_is_shared_with_run_loop(self):
        provider = ToolProvider()
        api_provider_registry.register(provider)
        model = Model(id="fake-runtime-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model), tools=[RuntimeWritingTool()])
        runtime = RuntimeState()
        runtime.set_variable("seed", "initial")
        agent.set_runtime(runtime)

        async for _event in agent.prompt(Message(role="user", content="write runtime")):
            pass

        self.assertIn("agent_value", agent.state.runtime.variables)
        self.assertEqual(agent.state.runtime.variables["agent_value"].raw_value, "omega")
        self.assertIn("seed", agent.state.runtime.variables)

    async def test_agent_accepts_injected_python_runtime(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="python-runtime-model", provider="fake", api=provider.api)
        runtime = IPythonProgramRuntime(initial_namespace={"seed": "python"})
        agent = Agent(AgentConfig(model), python_runtime=runtime)

        self.assertIs(agent.state.python_runtime, runtime)

        replacement_runtime = IPythonProgramRuntime(initial_namespace={"seed": "replacement"})
        agent.set_python_runtime(replacement_runtime)
        self.assertIs(agent.state.python_runtime, replacement_runtime)

    async def test_agent_bridges_model_registry_to_get_api_key(self):
        class FakeRegistry:
            def get_api_key(self, provider):
                return f"{provider}-registry-key"

        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="registry-key-model", provider="fake", api=provider.api)
        config = AgentConfig(model)
        config.model_registry = FakeRegistry()
        agent = Agent(config)

        async for _event in agent.prompt(Message(role="user", content="hello")):
            pass

        self.assertEqual(provider.last_api_key, "fake-registry-key")

    async def test_agent_prefers_explicit_api_key_resolver(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="resolver-key-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))
        agent.set_api_key_resolver(lambda provider_name: f"{provider_name}-resolver-key")

        async for _event in agent.prompt(Message(role="user", content="hello")):
            pass

        self.assertEqual(provider.last_api_key, "fake-resolver-key")

    async def test_agent_can_set_static_api_key(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="static-key-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model), api_key="static-key")

        async for _event in agent.prompt(Message(role="user", content="hello")):
            pass

        self.assertEqual(provider.last_api_key, "static-key")

    async def test_agent_setters_sync_runtime_config_view(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="setter-model", provider="fake", api=provider.api)
        config = AgentConfig(model)
        agent = Agent(config)

        replacement_model = Model(id="replacement-model", provider="fake", api=provider.api)
        agent.set_system_prompt("system prompt")
        agent.set_thinking_level("high")
        agent.set_model(replacement_model)

        self.assertEqual(agent.state.system_prompt, "system prompt")
        self.assertEqual(agent.state.thinking_level, "high")
        self.assertEqual(agent.state.model, replacement_model)
        self.assertEqual(config.system_prompt, "system prompt")
        self.assertEqual(config.thinking_level, "high")
        self.assertEqual(config.model, replacement_model)

    async def test_agent_propagates_stream_options_to_provider(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="options-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))
        agent.set_system_prompt("system prompt")
        agent.set_thinking_level("medium")
        agent.set_temperature(0.25)
        agent.set_max_tokens(222)
        agent.set_session_id("session-123")
        agent.set_transport("websocket")
        agent.set_cache_retention("short")
        agent.set_headers({"x-test": "1"})
        agent.set_max_retry_delay_ms(9000)
        agent.set_request_metadata({"trace_id": "trace-1"})
        agent.set_thinking_budgets({"medium": 321})
        agent.set_on_payload(lambda payload, _model: payload)

        async for _event in agent.prompt(Message(role="user", content="hello")):
            pass

        self.assertIsNotNone(provider.last_options)
        self.assertEqual(provider.last_options.system_prompt, "system prompt")
        self.assertEqual(provider.last_options.thinking_level, "medium")
        self.assertEqual(provider.last_options.temperature, 0.25)
        self.assertEqual(provider.last_options.max_tokens, 222)
        self.assertEqual(provider.last_options.session_id, "session-123")
        self.assertEqual(provider.last_options.transport, "websocket")
        self.assertEqual(provider.last_options.cache_retention, "short")
        self.assertEqual(provider.last_options.headers, {"x-test": "1"})
        self.assertEqual(provider.last_options.max_retry_delay_ms, 9000)
        self.assertEqual(provider.last_options.metadata, {"trace_id": "trace-1"})
        self.assertEqual(provider.last_options.thinking_budgets, {"medium": 321})
        self.assertIsNotNone(provider.last_options.on_payload)

    async def test_agent_exports_session_context_with_host_resources(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="session-export-model", provider="fake", api=provider.api)
        runtime = RuntimeState()
        python_runtime = IPythonProgramRuntime(initial_namespace={"seed": "python"})
        agent = Agent(
            AgentConfig(model),
            messages=[Message(role="user", content="hello")],
            runtime=runtime,
            python_runtime=python_runtime,
            api_key="session-key",
        )

        session_context = agent.export_session_context()

        self.assertIsInstance(session_context, AgentSessionContext)
        self.assertIs(session_context.host.runtime, runtime)
        self.assertIs(session_context.host.python_runtime, python_runtime)
        self.assertEqual(session_context.host.api_key, "session-key")
        self.assertEqual(session_context.messages[0].content, "hello")

    async def test_agent_can_replace_session_context_for_handoff(self):
        provider = FakeProvider()
        api_provider_registry.register(provider)
        model = Model(id="session-import-model", provider="fake", api=provider.api)
        agent = Agent(AgentConfig(model))

        handoff_runtime = RuntimeState()
        handoff_runtime.set_variable("seed", "handoff")
        handoff_python_runtime = IPythonProgramRuntime(initial_namespace={"handoff": True})
        host_context = AgentHostContext(
            runtime=handoff_runtime,
            python_runtime=handoff_python_runtime,
            api_key="handoff-key",
        )
        session_context = AgentSessionContext(
            messages=[Message(role="user", content="continue from another agent")],
            tools=[],
            host=host_context,
        )

        agent.replace_session_context(session_context)

        self.assertEqual(agent.state.messages[0].content, "continue from another agent")
        self.assertIs(agent.state.runtime, handoff_runtime)
        self.assertIs(agent.state.python_runtime, handoff_python_runtime)
        self.assertEqual(getattr(agent.config, "api_key", None), "handoff-key")

    async def test_agent_handoff_to_model_transforms_messages_and_keeps_host_resources(self):
        source_model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        target_model = Model(id="claude-4-6-sonnet-20241022", provider="anthropic", api="anthropic-messages")
        runtime = RuntimeState()
        runtime.set_variable("seed", "present")
        python_runtime = IPythonProgramRuntime(initial_namespace={"seed": "python"})
        agent = Agent(AgentConfig(source_model), runtime=runtime, python_runtime=python_runtime)
        assistant = AssistantMessage(
            content_blocks=[
                ThinkingBlock(thinking="private plan"),
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="dataset",
                            version=2,
                            summary_blocks=[TextBlock(text="10 rows ready")],
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
        agent.replace_messages([Message(role="user", content="hi"), assistant, tool_result])

        agent.handoff_to_model(target_model)

        self.assertIs(agent.state.runtime, runtime)
        self.assertIs(agent.state.python_runtime, python_runtime)
        self.assertEqual(agent.state.model, target_model)
        self.assertEqual(len(agent.state.messages), 1)
        self.assertEqual(agent.state.messages[0].content, "hi")

    async def test_agent_handoff_to_model_can_preserve_provider_state_when_explicitly_enabled(self):
        model = Model(id="gpt-5.4", provider="openai", api="openai-chat")
        agent = Agent(AgentConfig(model))
        assistant = AssistantMessage(
            content_blocks=[ThinkingBlock(thinking="keep this replay block", signature="sig-1")],
            provider_state={"openai": {"response_id": "resp_1"}},
            model=model.id,
            provider=model.provider,
            api=model.api,
        )
        agent.replace_messages([Message(role="user", content="hi"), assistant])

        agent.handoff_to_model(model, preserve_provider_state=True)

        transformed_assistant = agent.state.messages[1]
        self.assertEqual(transformed_assistant.provider_state["openai"]["response_id"], "resp_1")
        self.assertIsInstance(transformed_assistant.content_blocks[0], ThinkingBlock)


if __name__ == "__main__":
    unittest.main()
