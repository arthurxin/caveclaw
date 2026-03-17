# CaveClaw TODO

## 当前状态

### 已完成

- [x] 将消息协议升级到 block-based 主路径，同时保留旧 `content` 调用的兼容访问器。
- [x] 为 `AssistantMessage` 增加 `raw_content`，给 provider replay 留出统一协议位。
- [x] 引入 `RuntimeState / RuntimeVariable / RuntimeDeltaOp` 基础骨架。
- [x] 为 `agent_loop` 补上 `message_start / message_update / message_end`。
- [x] 为工具执行链路补上 `tool_execution_update`、`signal` 接口位和 `runtime_ops` 入口。
- [x] 为 `Agent` 补上基础控制面：`continue_run()`、队列模式、`wait_for_idle()`。
- [x] 为 `max_rounds` 场景补上轻量级 `handle_consolidation()` hook。
- [x] 补齐 Azure OpenAI Responses provider：
  支持 `instructions + input`、reasoning effort、原生 function calling、`previous_response_id` 续轮和 `function_call_output` 回传。
- [x] 完成 Azure provider 的真实 tool-calling smoke test。
- [x] 将 demo 共享的 provider 注册 / provider 构造逻辑抽到 `examples/provider_demos/demo_shared.py`。
- [x] 将 resolver 默认模型更新为当前配置一致的 `openai/gpt-5.4`。

## 当前优先级

### P0: 把 runtime 驱动 loop 做完整

- [x] 将 `agent_loop` 改成真正的 runtime 驱动主循环：
  一轮开始先注入 runtime snapshot，工具批次结束后统一 commit `runtime_ops`，生成新 snapshot 和短 worklog，再决定是否继续当前内层工具循环；当内层结束后，带着新的 snapshot 和 worklog 进入下一次外层审查。
- [x] 将 runtime snapshot 注入策略收紧为“只在需要重新交给模型思考之前注入”：
  避免重复注入无变化状态，但允许工具批次结束后立即生成新的世界状态投影。
- [x] 给 worklog 设计正式 block/tag：
  它表示工作轨迹，会进入后续 loop 的初始信息，但不应污染 UI-only 日志。

### P0.5: 把 runtime 变量语义补清楚

- [x] 为工具/技能增加变量声明伪接口：
  例如 `reads / writes / temp_outputs`，用于说明该工具想读取哪些变量、产生哪些中间变量和输出变量。
- [x] 完成 `RuntimeState.apply_ops()` 的规范化实现：
  把 `set / delete / merge / append / touch / replace_blocks` 收口到统一提交路径。
- [ ] 细化 `RuntimeVariable.kind` 的处理策略：
  `opaque / structured / tabular / message_blocks / binary_ref` 的 inspector 行为要不同。

### P1: 把 compaction 变成每次发给 provider 前的可扩展步骤

- [x] 明确 compaction 的职责边界：
  它主要发生在 `AgentMessage -> llm_provider message` 之间，负责去掉不该给 LLM 的 UI/log 富文本，压缩工作轨迹，并控制 runtime snapshot 的可见粒度。
- [x] 把 compaction 做成可扩展模块：
  现在先做“过滤 UI-only / log-only blocks + 裁剪 runtime metadata”的基础版，之后再扩展为更复杂的上下文压缩策略。
- [ ] 为 compaction 留下扩展 TODO：
  未来支持 token budget、变量级摘要策略、不同 provider 的差异化压缩规则。

### P1.5: 把 provider 适配层做稳

- [ ] 统一 provider 的工具调用、reasoning 事件和 usage 输出格式。
- [ ] 把 `minimax_local` 的特殊 replay 语义完全封装在 provider 内：
  runtime 在 `AgentMessage` 之上，provider 在 `AgentMessage` 之下，统一层仍然是 `AgentMessage`。
- [x] 引入 provider codec 基础层：
  让 `raw_content / provider_state` 的解释权逐步下沉到 provider 适配层，而不是继续堆在 loop 里。
- [x] 为 Gemini provider 补齐 `thoughtSignature` replay 和连续 tool result 合并逻辑。
- [x] 为 Volcengine provider 修复 tool call replay 的 arguments 字符串化问题。
- [x] 为 Azure provider 建立 codec / replay / continuation 基础路径。
- [ ] 给 `ModelRegistry` 增加更清楚的加载错误与配置校验。
- [ ] 给 Azure Responses provider 补 usage、streaming 增量和 provider_state 精细化映射。

### P2: 暂时只记 TODO，不急着实现

- [ ] UI runtime 面板：
  在涉及 `RuntimeVariable` 时，UI 显示更用户友好的可视化信息，而不是直接给 LLM 的 metadata 摘要。
- [ ] 子智能体 runtime 隔离：
  subagent 默认拥有独立 runtime，之后再决定如何继承父级快照或回写父级结果。
- [ ] 工具/技能变量编排：
  当前只补了 `reads / writes / temp_outputs` 和 runtime selection 伪接口；真正的 tool calling / variable orchestration 之后按自定义方案继续写，不走传统兼容式实现。

## 暂不优先

- [ ] 为了“看起来更聪明”而把太多业务逻辑塞进核心层。
- [ ] 过早做 DAG / 并发编排。
- [ ] 在没有稳定测试前继续快速扩 provider 数量。
