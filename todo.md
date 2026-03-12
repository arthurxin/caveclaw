# CaveClaw AgentCore 重构计划 (TODO)
> 基于 `agent_core` (异步流式引擎) 与 `src` (业务双层循环架构) 的融合设计。

## Phase 1: 核心生命周期与骨架重构 (**已完成**)
- ✅ **层次化 AgentEvent 重构**: 在 `types.py` 中为 `AgentEvent` 增加了 `event_id` 和 `parent_id` 属性，支持前端渲染酷炫的子树形进度流。
- ✅ **配置防线引入**: 在 `.env` 及 `agent_loop.py` 中引入了 `max_rounds` 和 `max_consecutive_tool_failures` 机制。
- ✅ **内联式 (Inline) 引擎重写**: 将工具执行合并进了主状态机 `run_loop` 中，让大模型思维步骤可以随时 `yield` 最新的一手内部消息。
- ✅ **熔断与求援钩子**: 实装了达到条件时自动抛出 `consolidation_required` (请应用层处理记忆浓缩) 和 `human_intervention_required` (死活配不对工具时的求救)。

## Phase 4: 万能 LLM 包装器 (`Universal LLM Wrapper`) (**已完成**)
- ✅ **模型与厂商定义**: 复刻了 `api_registry` 和 `providers` 体系，隔离了供应商 SDK。
- ✅ **动态模型解析 (`ModelResolver`)**: 支持 `provider/model_id:thinking_level` 格式的智能路由。
- ✅ **多厂商支持**: 实装了 OpenAI (及兼容接口), Anthropic, Google Gemini 以及 MiniMax (带 Think 块剥离)。
- ✅ **引擎原生接入**: `agent_loop.py` 已支持通过 `ApiProvider` 动态派发，`stream_fn` 已变为可选。

## Phase 5: 智能基础设施与工程化 (计划中)
> 借鉴 OpenClaw (pi-mono) 的工程化实践，优化 Agent 的“智商”与“体感”。
- [ ] **动态提示词管理器 (`SystemPromptBuilder`)**: 实现模块化 Prompt 拼接，根据当前工具和环境自动裁剪指令，节省 Token。
- [ ] **技能延迟加载 (`Skill Registry & Lazy Loading`)**: 
    - 引入 `skills/` 目录规范，Prompt 里只放简介。
    - 增加 `read_skill` 工具，让 AI 自主按需读取详细手册。
- [ ] **输出抑制 (`NO_REPLY`)**: 在 `agent_loop` 层面处理静默回复令牌，杜绝无意义的对话生成。

## Phase 6: 企业级增强与自主性 (愿景)
- [ ] **子智能体协作 (`Subagent Orchestration`)**: 支持 `sessions_spawn`，让 Agent 能够派生小号并行处理子任务。
- [ ] **长效记忆系统 (`Long-term Memory`)**: 实现 `shared_memory` 的文件持久化与 RAG 检索。
- [ ] **人工审批流 (`Human-in-the-loop`)**: 针对高危工具（如删除、部署）增加 AgentEvent 级别的中断确认机制。
- [ ] **标准化推理事件**: 统一所有厂商的 CoT (Think 块) 吞吐协议。

---

## ⏸ 暂停/延后优化的功能 (Suspended / Backlog)
1. **[暂停] 人工逐行审核 (Blocking Steering Phase)**
   - 原因讨论: 让用户每调一个工具就人工确认一次太花人力/时间。
   - 妥协点: 修改为了仅日志记录 (Observation-only Steering)。
2. **[暂停] 动态并发调度机制 (Dependency-Aware Execution & DAG)**
   - 纳入 Phase 6 的并行化考量，暂不作为底层强制要求。
