import unittest

from agent_core.compaction import compact_messages_for_llm
from agent_core.types import AgentContext, Message, RuntimeSnapshotBlock, RuntimeSnapshotEntry, TextBlock


class CompactionConfig:
    async def compact_messages(self, messages, context):
        return messages


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_compaction_drops_ui_only_blocks_and_messages(self):
        messages = [
            Message(role="system", content_blocks=[TextBlock(text="hidden", metadata={"ui_only": True})]),
            Message(role="user", content_blocks=[TextBlock(text="visible")]),
            Message(role="assistant", content="skip me", metadata={"exclude_from_llm": True}),
        ]

        compacted = await compact_messages_for_llm(messages, AgentContext(messages=[]), CompactionConfig())

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].content, "visible")

    async def test_compaction_keeps_runtime_snapshot_but_trims_metadata(self):
        snapshot_message = Message(
            role="system",
            content_blocks=[
                RuntimeSnapshotBlock(
                    entries=[
                        RuntimeSnapshotEntry(
                            key="artifact",
                            version=2,
                            summary_blocks=[TextBlock(text="artifact summary")],
                            metadata={"display_name": "Artifact", "ui": {"color": "red"}},
                        )
                    ]
                )
            ],
        )

        compacted = await compact_messages_for_llm([snapshot_message], AgentContext(messages=[]), CompactionConfig())

        runtime_block = compacted[0].content_blocks[0]
        self.assertIsInstance(runtime_block, RuntimeSnapshotBlock)
        self.assertEqual(runtime_block.entries[0].metadata, {"display_name": "Artifact"})


if __name__ == "__main__":
    unittest.main()
