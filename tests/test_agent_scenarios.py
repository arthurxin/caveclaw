import asyncio
from typing import AsyncGenerator, Dict, Any, List
import uuid

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_core.types import (
    AgentTool, ToolResult, AgentContext, AgentEvent, ToolCall, AssistantMessage, AgentLoopConfig,
    AgentMessage, Message, ToolResultMessage
)
from agent_core.agent_loop import run_loop
from agent_core.inspector import PythonRuntimeInspector


# ==========================================
# 1. 基础构件准备：配置、工具、以及辅助打印函数
# ==========================================

class DemoConfig(AgentLoopConfig):
    """
    一个简单的配置类，展示如何设置拦截阈值。
    """
    # 演示：设置极小的限制以快速触发保护机制
    max_rounds = 3
    max_consecutive_tool_failures = 2

    async def convert_to_llm(self, messages: List[AgentMessage]) -> List[Message]:
        return messages

    async def transform_context(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        return messages

    async def get_steering_messages(self) -> List[AgentMessage]:
        return []

    async def get_followup_messages(self) -> List[AgentMessage]:
        return []

class DatabaseQueryTool(AgentTool):
    """一个模拟查询数据库的 Context-Aware 工具"""
    def __init__(self):
        super().__init__("query_db", "查询数据库表", {"sql": "string"}, "DB Query")

    async def execute(self, tool_call_id: str, params: Dict[str, Any], context: AgentContext, on_update=None) -> ToolResult:
        sql = params.get("sql", "")
        if "DROP" in sql.upper():
            raise ValueError("Permission Denied: Cannot drop tables.")
            
        if sql == "SELECT * FROM users":
            # 正常查询，带回一个庞大的内部状态
            fake_data = {"count": 1000, "sample": ["Alice", "Bob", "Charlie"]}
            return ToolResult(
                content=f"查询成功，找到 {fake_data['count']} 条记录。",
                state_delta={"last_query_result": fake_data} # 注入环境增量
            )
        
        return ToolResult(content="查询语句为空或无法识别。")


async def run_scenario(name: str, stream_fn) -> AgentContext:
    """包装执行，用于捕获进度并打出漂亮的 UI 事件树"""
    print(f"\n{'='*50}")
    print(f"🎬 Scenario: {name}")
    print(f"{'='*50}\n")
    
    config = DemoConfig()
    tools = [DatabaseQueryTool()]
    
    # 模拟外部一直维护着这个对话上下文
    context_msgs = [Message(role="user", content="请根据情景描述完成任务。")]
    
    # 获取迭代器
    loop_gen = run_loop(context_msgs, tools, config, stream_fn)
    
    # 这里是未来放在前端或者 CLI 的事件监听器
    try:
        async for event in loop_gen:
            if event.type == "turn_start":
                print(f"[🔄 思考回合 {event.data['round']} 开始]")
                
            elif event.type == "tool_execution_start":
                tc = event.data['tool_call']
                print(f"   🛠️  调用工具: {tc.name}({tc.arguments})")
                
            elif event.type == "tool_execution_success":
                print(f"   ✅ 工具成功: {event.data['result']}")
                if event.data.get('delta_applied'):
                    print(f"   💾 (环境状态已自动合并至 shared_memory)")
                    
            elif event.type == "tool_execution_error":
                print(f"   ❌ 工具报错: {event.data['error']}")
                
            elif event.type == "human_intervention_required":
                print(f"\n🚨 [熔断保护触发] 连续失败 {event.data['consecutive_failures']} 次，已大呼救！")
                
            elif event.type == "consolidation_required":
                print(f"\n🧠 [记忆浓缩触发] 突破单次 {event.data['max_rounds']} 回合限制，准备触发 Phase 3...")
                
            elif event.type == "turn_end":
                print(f"[🏁 思考回合结束] 是否有工具执行: {event.data.get('tools_executed', 0) > 0}")
                
    except Exception as e:
        print(f"框架异常退出: {e}")

    # 给未来的 Phase 3 (Inspector) 准备最终上下文
    # 注意：在真实的体系里，agent 实例会直接持有这套 context，这里由于直接调 run_loop，
    # 我们根据执行结果虚拟出一个包含最终 shared_memory 的 Context 来展示 Inspector 效果。
    fake_final_context = AgentContext(messages=context_msgs)
    
    return fake_final_context


# ==========================================
# 2. 编写模拟测试剧情 (Mock Streams)
# ==========================================

async def scenario_1_normal_success_no_tools(messages):
    """情景 1：大模型直接回答问题，不需要任何工具"""
    # 直接 yield 文本内容，没有 tool_calls
    yield {"content": "这是一次正常的回答，我不需要查数据库就能告诉你 Alice 是系统管理员。"}


async def scenario_2_success_with_state_delta(messages):
    """情景 2：调用查询工具成功，并注入环境状态"""
    # 模拟模型输出工具调用
    yield {
        "tool_calls": [
            ToolCall(id="call_1", name="query_db", arguments={"sql": "SELECT * FROM users"}).__dict__
        ]
    }


async def scenario_3_consecutive_failures_fuse(messages):
    """情景 3：大模型像无头苍蝇一样，连着两次写入错误的 SQL 语法，触发熔断"""
    # 不管是哪一轮，这个假模型都会坚决要删除表 (触发我们的 Permission Denied Exception)
    yield {
        "tool_calls": [
            ToolCall(id=str(uuid.uuid4()), name="query_db", arguments={"sql": "DROP TABLE users;"}).__dict__
        ]
    }


async def scenario_4_max_rounds_consolidation(messages):
    """情景 4：大模型陷入查错死循环，但每次的报错不同（比如被工具骗了），最终触及回合上限"""
    # 这个假模型会每次查一个不存在的表，每次成功，但找不到结果，于是继续下一回合，永远停不下来
    yield {
        "tool_calls": [
            ToolCall(id=str(uuid.uuid4()), name="query_db", arguments={"sql": "SELECT * FROM ghosts"}).__dict__
        ]
    }

# ==========================================
# 3. 运行测试
# ==========================================

async def main():
    print("洞穴之爪 (CaveClaw) AgentCore 行为展示\n")
    
    # 演示 1
    await run_scenario("正常直接回答", scenario_1_normal_success_no_tools)
    
    # 演示 2
    ctx2 = await run_scenario("调用工具并变更 Shared Memory", scenario_2_success_with_state_delta)
    # 因为跑测试的时候没有真实绑定共享内存引用，我们在外面假装手动运行一次来看 Inspector 的抓取能力
    tool = DatabaseQueryTool()
    await tool.execute("id", {"sql": "SELECT * FROM users"}, ctx2)
    ctx2.shared_memory.update({"last_query_result": {"count": 1000, "sample": ["Alice", "Bob"]}}) 
    
    print("\n--- Phase 3: Runtime Inspector 抓取状态 ---")
    inspector = PythonRuntimeInspector()
    print(await inspector.capture_state(ctx2))
    
    # 演示 3：熔断保护
    await run_scenario("连环报错熔断 (防 API 爆炸)", scenario_3_consecutive_failures_fuse)
    
    # 演示 4：死循环拦截
    await run_scenario("死循环拦截 (强制归档)", scenario_4_max_rounds_consolidation)


if __name__ == "__main__":
    asyncio.run(main())
