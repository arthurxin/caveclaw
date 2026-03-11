import asyncio
from typing import AsyncGenerator, Dict, Any, List
import traceback

# Setup sys path so we can import from caveclaw directly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_core.types import (
    AgentTool, ToolResult, AgentContext, AgentEvent, ToolCall, AssistantMessage, AgentLoopConfig,
    AgentMessage, Message
)
from agent_core.agent_loop import run_loop
from agent_core.inspector import PythonRuntimeInspector

# 1. Create a Fake Config
class MockConfig(AgentLoopConfig):
    max_rounds = 3
    max_consecutive_tool_failures = 3

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        return messages

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        return messages

    async def get_steering_messages(self) -> List[AgentMessage]:
        return []

    async def get_followup_messages(self) -> List[AgentMessage]:
        return []

# 2. Create a Mock Tool that returns state_delta
class DataFetchTool(AgentTool):
    def __init__(self):
        super().__init__("fetch_data", "Fetch fake data", {}, "Fetch Data")

    async def execute(self, tool_call_id: str, params: Dict[str, Any], context: AgentContext, on_update=None) -> ToolResult:
        # Simulate fetching a large payload (like a dataframe, represented here as a dict)
        large_payload = {"user_1": "Alice", "user_2": "Bob", "count": 2, "rows": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        
        return ToolResult(
            content="Data fetched successfully.",
            state_delta={"recent_users_table": large_payload}
        )

# 3. Create a Mock LLM Stream that always invokes the tool
async def mock_stream_fn(messages) -> AsyncGenerator[Any, None]:
    # We yield a tool call chunk to trigger the loop
    yield {"tool_calls": [{"id": "call_123", "name": "fetch_data", "arguments": {}}]}

# 4. Main test logic
async def main():
    config = MockConfig()
    tools = [DataFetchTool()]
    initial_messages = [Message(role="user", content="Go fetch the data")]
    
    print("--- Starting Agent Loop ---")
    final_messages = []
    
    # Run loop strictly for 1 iteration by breaking manually or letting max_rounds hit
    loop_gen = run_loop(initial_messages, tools, config, mock_stream_fn)
    
    try:
        async for event in loop_gen:
            print(f"Event Triggered: {event.type}")
            if event.type == "tool_execution_success":
                print(f"  -> {event.data}")
            elif event.type == "agent_end":
                final_messages = event.data["messages"]
            elif event.type == "consolidation_required":
                print(f"  -> 达到轮次上限, 引擎请求 Phase 3: Consolidation")
                break
    except Exception as e:
        print(f"Loop Exception:")
        traceback.print_exc()

    print("\n--- Phase 3: Inspecting Captured State (Environment Inspector) ---")
    
    # After the loop breaks/ends, we grab the final context from the first run.
    # Note: run_loop does not return the `AgentContext` directly to avoid breaking old interfaces.
    # In a real app with StateConsolidator, `run_loop` would invoke `on_consolidate(messages, context)`
    
    # We will instantiate a new Inspector and simulate what the Phase 3 hook would see 
    # based on the final_messages we printed.
    
    # Actually, to demonstrate the shared_memory mutation, let's manually invoke the tool and inspector:
    dummy_ctx = AgentContext(messages=[], shared_memory={})
    print(f"Initial State: {dummy_ctx.shared_memory}")
    
    res = await tools[0].execute("test", {}, dummy_ctx)
    dummy_ctx.shared_memory.update(res.state_delta)
    
    inspector = PythonRuntimeInspector()
    report = await inspector.capture_state(dummy_ctx)
    
    print("\n[Inspector Report to be sent back to LLM for review]")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
