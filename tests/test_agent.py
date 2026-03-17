import unittest

from agent_core import Agent, Message
from agent_core.assistant_messages import AgentContext, AgentTool, RuntimeState, ToolResult
from agent_core.llm_provider import Model, api_provider_registry


class FakeProvider:
    api = "fake-agent-provider"

    def __init__(self):
        self.calls = 0

    async def stream(self, model, messages, options, api_key=None):
        self.calls += 1
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


if __name__ == "__main__":
    unittest.main()
