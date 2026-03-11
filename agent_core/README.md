# CaveClaw AgentCore

AgentCore 是一个纯粹、底层、解耦的大语言模型 (LLM) 代理心跳引擎。
它融合了双重循环设计原则与响应式事件流机制，为上层应用提供高度可观测、安全的工具执行环境。

由于这是一个在 [OpenClaw](https://github.com/) 原有 `agent_core` 之上进行深度重构的版本，老的文档已归档至 [`openclaw_agent_loop_readme.md`](./openclaw_agent_loop_readme.md) 供参考比对。

---

## 核心设计理念

AgentCore 的设计目标是：**不绑定任何具体业务逻辑或提示词格式，只做最严密的安全验证和上下文流转分发**。

它摒弃了将巨量执行数据直接塞给 LLM 或修改全局环境的做法，引入了**环境检查器 (Environment Inspector)** 和**状态增量 (State Delta)** 架构。它将“智体思考”与“世界状态”清楚地分割开来。

## 目录结构
- `types.py`: 提供高度结构化的数据规约 (Protocol)，定义了诸如 `AgentContext`, `ToolResult`, `AgentEvent` 以及生命周期钩子 `AgentLoopConfig`。
- `agent_loop.py`: 核心状态机 `run_loop`。采用 `yield` 形式向外透明地暴露异步的层次化事件节点。
- `agent.py`: 维护上层队列状态，提供人工打断 (Steering) 和队列追加 (Follow-up) 的控制句柄。
- `inspector.py`: （新增）提供底层运行环境的数据抓取与安全浓缩机制 (`StateReducer`)。

---

## 架构亮点

### 1. 深度可观测的进度树 (Hierarchical Agent Events)
在 `run_loop` 中的每一步迭代，都会向外抛出包含 `event_id` 和 `parent_id` 的结构化 `AgentEvent` 事件。
UI 前端可借此画出极为优美的任务拆解树，例如：
```
└─ agent_start (id: A)
   ├─ turn_start (id: B, parent: A)
   │  ├─ tool_execution_start (id: C, parent: B, tool: SQLQuery)
   │  └─ tool_execution_success (id: D, parent: C)
   └─ turn_end (parent: B)
```

### 2. 底线保护熔断器 (Fail-Safe Quotas)
为防止 AI陷入不可控的无限生成或烧费：
- **`MAX_ROUNDS`**: 单次推理的最大往复轮数。一旦超过（如发生找数据的死循环），触发 `consolidation_required` 事件，通知外部触发记忆归档系统（Phase 3）。
- **`MAX_CONSECUTIVE_FAILURES`**: 连环纠错容忍度。如果模型连续不断地执行报错指令（如写错 5 次 SQL），立即触发 `human_intervention_required` 人工求助信号。

### 3. 环境状态减噪器 (Context-Aware Tools & Reducers)
- **`State Delta`**: 工具层只能通过返回 `ToolResult(content="成功", state_delta={...})` 来安全地更新环境。执行引擎会以合并字典的形式更新 `AgentContext.shared_memory`。
- **`PythonRuntimeInspector`**: 在面对动辄几 GB 或几万行的 DataFrame / List 时，自动挂载的 `Reducer` (如 PandasReducer) 会将其“降噪压缩”为包含 `shape`、`memory_usage` 及 `head(3)` 的无害字符串摘要，彻底避免模型 Context Window 爆炸。

---

## 示例用法 (Quick Start)

定义你的 Config 和 Context-Aware Tool：

```python
import asyncio
from typing import Dict, Any
from agent_core.types import AgentTool, ToolResult, AgentContext
from agent_core.inspector import PythonRuntimeInspector

class FakeDBTool(AgentTool):
    def __init__(self):
        super().__init__("query_db", "查询用户库", {"sql": "str"}, "Query")

    async def execute(self, tool_call_id: str, params: Dict[str, Any], context: AgentContext, on_update=None) -> ToolResult:
        # 你可以从外部取到之前积累的安全上下文
        last_table_name = context.shared_memory.get("last_queried_table", "users")
        
        # 不要把10万条数据转成字符串，塞进 state_delta 里
        huge_data_struct = {"users": [{"name": "Auth"} for _ in range(100000)]}
        return ToolResult(
            content="查询完成，详情见上下文环境。",
            state_delta={"last_queried_table": "users", "raw_data": huge_data_struct}
        )
```

当你需要大模型在下一轮感知这些数据时，通过 `PythonRuntimeInspector` 提纯状态给大模型：
```python
inspector = PythonRuntimeInspector()
report = await inspector.capture_state(agent_context)
print(report) 
# LLM 看到的只有： Dict with 10 keys: ... <Truncated> 
# 完美避开了长度超标错误。
```

> **Roadmap:** 未来将在外部集成 `Planning-with-Files` 模块，通过结合外设磁盘实现跨任务的 Multi-Agent 文件流转。
