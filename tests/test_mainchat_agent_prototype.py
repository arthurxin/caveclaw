import unittest

from Agent_Prototype.mainchat_agent import MainChatAgent
from agent_core import Message


class ConcreteMainChatAgent(MainChatAgent):
    def build_system_prompt(self) -> str:
        return "You are a concrete main chat test agent."


async def fake_stream(messages):
    last_user_message = next(
        (message for message in reversed(messages) if getattr(message, "role", None) == "user"),
        None,
    )
    text = last_user_message.content if last_user_message is not None else ""
    yield {"content": f"ack: {text}"}


class MainChatAgentPrototypeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mainchat_agent_requires_concrete_subclass(self):
        with self.assertRaises(TypeError):
            MainChatAgent()

    async def test_concrete_mainchat_agent_runs_single_prompt(self):
        agent = ConcreteMainChatAgent()
        assistant_messages = []

        async for event in agent.handle_user_input("analyze this dataframe", stream_fn=fake_stream):
            if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
                assistant_messages.append(event.message.content)

        self.assertEqual(agent.agent.state.messages[0].role, "user")
        self.assertEqual(agent.agent.state.messages[0].content, "analyze this dataframe")
        self.assertEqual(assistant_messages, ["ack: analyze this dataframe"])
        self.assertEqual(agent.agent.state.messages[-1].role, "assistant")

    async def test_build_user_message_creates_user_role_message(self):
        agent = ConcreteMainChatAgent()

        message = agent.build_user_message("hello")

        self.assertIsInstance(message, Message)
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "hello")


if __name__ == "__main__":
    unittest.main()
