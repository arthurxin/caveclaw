# CaveClaw AgentCore

AgentCore 是一个纯粹、底层、解耦的大语言模型 (LLM) 代理心跳引擎。
它融合了双重循环设计原则与响应式事件流机制，为上层应用提供高度可观测、安全的工具执行环境。

仓库级说明、目录总览和当前优先级请先看根目录 `README.md`。
旧版设计文档已归档到 `docs/archive/` 目录，供参考比对。

---

## 核心设计理念

AgentCore 的设计目标是：**不绑定任何具体业务逻辑或提示词格式，只做最严密的安全验证和上下文流转分发**。

它摒弃了将巨量执行数据直接塞给 LLM 或修改全局环境的做法，引入了**环境检查器 (Environment Inspector)** 和**状态增量 (State Delta)** 架构。它将“智体思考”与“世界状态”清楚地分割开来。

## 目录结构
- `types.py`: 提供 block-based 消息协议、runtime 状态对象和工具接口定义。
- `agent_loop.py`: 主循环编排层，只负责 turn/tool/runtime 的流程控制。
- `assistant_stream.py`: 负责把 provider chunk 累积成 `AssistantMessage`，并发出 `message_start/update/end`。
- `tool_execution.py`: 负责工具调用、`partial update` 透传、取消信号接入以及 staged `runtime_ops` 收集。
- `runtime_projection.py`: 负责 runtime snapshot、worklog、runtime commit 这些“runtime -> message”的投影逻辑。
- `compaction.py`: 负责把 `AgentMessage` 压缩成适合发给 provider 的上下文，过滤 UI-only / log-only 信息。
- `agent.py`: 维护上层队列状态，提供 steering、follow-up、continue、abort、wait_for_idle 等控制句柄。
- `inspector.py`: 提供底层运行环境的数据抓取与安全浓缩机制 (`StateReducer`)。

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

### 3. Runtime 投影与状态减噪
- **`RuntimeState`**: 工具不直接修改消息历史，而是通过 `state_delta` / `runtime_ops` 更新 runtime。
- **`Runtime Snapshot`**: loop 会在需要重新交给模型思考前，把当前 runtime 投影成一条 `AgentMessage`。
- **`Worklog`**: 每批工具调用结束后都会生成一条短工作轨迹，作为后续 loop 的上下文之一。
- **`Compaction`**: 每次真正发给 provider 前，都会先过滤 UI-only / log-only block，并裁剪 runtime metadata。
- **`PythonRuntimeInspector`**: 在面对动辄几 GB 或几万行的 DataFrame / List 时，Reducer 会把它们压成安全摘要，避免 Context Window 爆炸。

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
