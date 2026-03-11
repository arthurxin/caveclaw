# CaveClaw AgentCore 重构计划 (TODO)
> 基于 `agent_core` (异步流式引擎) 与 `src` (业务双层循环架构) 的融合设计。

## Phase 1: 核心生命周期与骨架重构 (**已完成**)
- ✅ **层次化 AgentEvent 重构**: 在 `types.py` 中为 `AgentEvent` 增加了 `event_id` 和 `parent_id` 属性，支持前端渲染酷炫的子树形进度流。
- ✅ **配置防线引入**: 在 `.env` 及 `agent_loop.py` 中引入了 `max_rounds` 和 `max_consecutive_tool_failures` 机制。
- ✅ **内联式 (Inline) 引擎重写**: 将工具执行合并进了主状态机 `run_loop` 中，让大模型思维步骤可以随时 `yield` 最新的一手内部消息。
- ✅ **熔断与求援钩子**: 实装了达到条件时自动抛出 `consolidation_required` (请应用层处理记忆浓缩) 和 `human_intervention_required` (死活配不对工具时的求救)。

## Phase 2: 上下文感知工具与运行环境探测器 (进行中)
- [ ] **重写工具签名 (`Context-Aware Tools`)**: 修改 `types.py` 中的 `AgentTool.execute()` 签名，让它的入参带上包含 `shared_memory` 的环境上下文 (`AgentContext`)。
- [ ] **增量式状态合并 (`State Delta`)**: 工具不再暴戾地自行修改整体环境，而是应该返回一个安全的可序列化的增量 `state_delta` 交由框架统一缝合。
- [ ] **Python Runtime 探测器接口 (`Environment Inspector`)**: 引入 `StateReducer` 基类和子类，允许在 Phase 3 时压缩如 大字典或超长类等状态。

## Phase 3: Planning-with-Files 与 架构集成 (计划中)
> 基于 Manus (Meta $2b 级别收购案例) 的 3-File Pattern. 引入磁盘持久化机制。
- [ ] **子任务文件化 (Inner Planning)**: 当触发多步大型工具使用前，在系统内生成独立的 `task_plan.md`, `findings.md`, `progress.md` 存放工作进度。
- [ ] **记忆整合器 (`StateConsolidator`)**: 实现一个外部 Hook，利用 Phase 1 预留的 `consolidation_required`，接收前面超长的对话栈并用强大的 LLM 返回一份精简压缩包，重启对话，杜绝幻觉死机。
- [ ] **集成测试案例**: 跑通一套“查询 -> 记录 -> 重审 -> 更新”的闭环测试系统。

---

## ⏸ 暂停/延后优化的功能 (Suspended / Backlog)
1. **[暂停] 人工逐行审核 (Blocking Steering Phase)**
   - 原因讨论: 让用户每调一个工具就人工确认一次太花人力/时间。
   - 妥协点: 修改为了仅日志记录 (Observation-only Steering)。除非检测到显式的危险工具，平时不做硬阻塞拦截。
   - 当前状态: 在现有设计中搁置。

2. **[暂停] 动态并发调度机制 (Dependency-Aware Execution & DAG)**
   - 原因讨论: 虽然 LLM 有一定能力判断哪些依赖能异步开多线程 (`asyncio.gather()`)跑，但实现太过超前，会大幅增加当前的底层调试难度。
   - 妥协点: 先确保框架和文件记忆（Planning-with-Files）主轴通顺稳定。
   - 当前状态: 记录为未来优化项（先依然保持序列式工具调用）。
