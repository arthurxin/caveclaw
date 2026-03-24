import tempfile
import unittest
from pathlib import Path

from Agent_Prototype.mainchat_agent import MainChatAgent, MainChatAgentConfig
from agent_core import Message
from agent_core.core.inspector import PythonRuntimeInspector


class DataframePrototypeAgent(MainChatAgent):
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        super().__init__()

    def build_system_prompt(self) -> str:
        return "Use python blocks for CSV analysis."

    def configure_config(self, config: MainChatAgentConfig) -> None:
        config.python_program_execution = True
        config.python_program_backend = "python"
        config.inspector = PythonRuntimeInspector()


class MainDataframeFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_mainchat_agent_can_read_csv_via_python_lane_and_bridge_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sales.csv"
            csv_path.write_text(
                "region,sales,orders\nNorth,120,4\nSouth,85,3\nEast,140,5\n",
                encoding="utf-8",
            )

            agent = DataframePrototypeAgent(csv_path)

            async def fake_stream(messages):
                has_runtime_snapshot = any(
                    isinstance(message, Message) and message.metadata.get("runtime_injected")
                    for message in messages
                )
                has_python_result = any(
                    isinstance(message, Message) and message.metadata.get("python_program_execution")
                    for message in messages
                )
                if has_runtime_snapshot and has_python_result:
                    yield {"content": "The top region is East."}
                    return

                yield {
                    "content": (
                        "```python\n"
                        "import pandas as pd\n"
                        f"sales_df = pd.read_csv(r'{csv_path}')\n"
                        "sales_summary = {'top_region': sales_df.sort_values('sales', ascending=False).iloc[0]['region']}\n"
                        "print(sales_summary['top_region'])\n"
                        "```"
                    )
                }

            final_assistant_messages = []
            synced_variables = None

            async for event in agent.handle_user_input(
                f"Analyze csv {csv_path}",
                stream_fn=fake_stream,
            ):
                if event.type == "python_program_execution_success":
                    synced_variables = event.get("synced_variables")
                if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
                    final_assistant_messages.append(event.message.content)

            self.assertIn("sales_df", agent.agent.state.runtime.variables)
            self.assertIn("sales_summary", agent.agent.state.runtime.variables)
            self.assertEqual(agent.agent.state.runtime.variables["sales_summary"].raw_value["top_region"], "East")
            self.assertEqual(synced_variables, ["sales_df", "sales_summary"])
            self.assertEqual(final_assistant_messages[-1], "The top region is East.")


if __name__ == "__main__":
    unittest.main()
