import unittest

from agent_core.assistant_messages import AgentContext, RuntimeSnapshotBlock, TextBlock
from agent_core.core.inspector import PythonRuntimeInspector


class InspectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_state_summarizes_long_lists(self):
        inspector = PythonRuntimeInspector()
        context = AgentContext(
            messages=[],
            shared_memory={
                "numbers": list(range(12)),
            },
        )

        report = await inspector.capture_state(context)

        self.assertIsInstance(report, RuntimeSnapshotBlock)
        self.assertEqual(report.entries[0].key, "numbers")
        summary_text = report.entries[0].summary_blocks[0]
        self.assertIsInstance(summary_text, TextBlock)
        self.assertIn("List (len=12)", summary_text.text)
        self.assertIn("more items", summary_text.text)

    async def test_capture_state_handles_large_dicts(self):
        inspector = PythonRuntimeInspector()
        payload = {f"key_{index}": index for index in range(15)}
        context = AgentContext(messages=[], shared_memory={"payload": payload})

        report = await inspector.capture_state(context)

        self.assertIsInstance(report, RuntimeSnapshotBlock)
        self.assertEqual(report.entries[0].key, "payload")
        summary_text = report.entries[0].summary_blocks[0]
        self.assertIsInstance(summary_text, TextBlock)
        self.assertIn("Dict with 15 keys", summary_text.text)
        self.assertIn("key_0", summary_text.text)

    async def test_capture_state_prefers_runtime_llm_view(self):
        inspector = PythonRuntimeInspector()
        context = AgentContext(messages=[])
        context.runtime.set_variable(
            "artifact",
            {"path": "/tmp/result.txt"},
            llm_view=[TextBlock(text="artifact ready")],
            metadata={"ui_color": "red", "display_name": "Artifact"},
        )

        report = await inspector.capture_state(context)

        self.assertEqual(report.entries[0].summary_blocks[0].text, "artifact ready")
        self.assertEqual(report.entries[0].metadata["display_name"], "Artifact")
        self.assertNotIn("ui_color", report.entries[0].metadata)

    async def test_capture_state_summarizes_dataframe_and_series(self):
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - defensive branch
            self.skipTest("pandas is not installed")

        inspector = PythonRuntimeInspector()
        df = pd.DataFrame(
            {
                "region": ["North", "East"],
                "sales": [120, 140],
            }
        )
        totals = df.groupby("region")["sales"].sum()
        context = AgentContext(messages=[], shared_memory={"df": df, "totals": totals})

        report = await inspector.capture_state(context)

        entries = {entry.key: entry for entry in report.entries}
        self.assertIn("df", entries)
        self.assertIn("totals", entries)
        df_text = entries["df"].summary_blocks[0]
        totals_text = entries["totals"].summary_blocks[0]
        self.assertIsInstance(df_text, TextBlock)
        self.assertIsInstance(totals_text, TextBlock)
        self.assertIn("DataFrame:", df_text.text)
        self.assertIn("Columns: ['region', 'sales']", df_text.text)
        self.assertIn("Series:", totals_text.text)
        self.assertIn("Dtype:", totals_text.text)


if __name__ == "__main__":
    unittest.main()
