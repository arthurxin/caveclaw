# Core Layer

`agent_core/core/` 放 CaveClaw 的执行引擎内核。

这里负责：

- `agent.py`
  `Agent` 高层控制面，维护会话级 `messages`、`runtime`、`python_runtime`，以及 steering / follow-up / continue / abort / wait_for_idle。
  也负责把宿主侧的 credential 解析桥接成 `get_api_key()` / `api_key` 注入给内核。
  这层应保持 Pythonic、薄控制面，不承载 UI 习惯、产品态 facade、或业务编排逻辑。
- `session_context.py`
  `AgentHostContext` / `AgentSessionContext`，把会话级宿主资源和可 handoff 的 session 包显式建模，供更上层 agent 传递。
- `handoff.py`
  底层 handoff substrate。负责把消息或 `AgentSessionContext` 转成目标 model/provider 可继续 replay 的形态，不依赖 `Agent` facade。
- `agent_loop.py`
  纯内存 runtime 驱动主循环，负责 turn、tool batch、runtime snapshot、worklog、结束条件。
- `tool_execution.py`
  工具执行、partial update、取消信号、staged runtime ops 收集。
- `python_program_execution/`
  `"""python"""` fenced block 的独立执行通道，和 native tool lane 并列，不混入 `tool_execution.py`。
- `runtime_projection.py`
  `RuntimeState -> AgentMessage` 的投影逻辑，包括 snapshot、worklog、runtime commit。
- `compaction.py`
  发送给 LLM 前的上下文裁剪与消息压缩。
- `inspector.py`
  runtime 变量摘要、Reducer、环境状态抓取。

这层不直接处理厂商 API 协议，也不承载 UI 逻辑，不在 import 时读取 `.env` 或环境变量。
credential 解析应发生在 host / Agent 侧，而不是 provider 侧。

## Execution Lanes

- **native tool lane** — 继续走 provider 原生 `tool_calls`，由 `tool_execution.py` 负责
- **python program lane** — 当 assistant 输出独立的 python block 且没有 native `tool_calls` 时，由 `python_program_execution/` 负责

两条 lane 是分开的，不会被伪装成同一种调用方式。

---

## 审阅建议

### ✅ 做得好的地方
- **`agent.py` 职责边界清晰**：只做控制面，不直接调 LLM，把 `run_loop` 委托给 `agent_loop.py`。
- **`agent_loop.py` 事件设计合理**：`turn_start / turn_end / agent_start / agent_end / tool_execution_*` 分层明确，易于上层订阅。
- **`handoff.py` 完全独立于 `Agent`**：可以直接调用 `handoff_messages()` 而不依赖高层 facade，非常干净。

### 🔧 建议改进

**`agent.py` — setter 代码重复**
每个 setter 都同时调用 `self.state.xxx = ...` 和 `setattr(self.config, "xxx", ...)` 共约 14 个 setter，模式完全相同。建议提取一个私有辅助：
```python
def _sync(self, key: str, value) -> None:
    setattr(self.state, key, value)
    setattr(self.config, key, value)
```
这样每个 setter 缩减为一行，可读性大幅提升。

**`agent.py` — `__init__` 过长**
`__init__` 约 50 行，初始化 `AgentState`、`AgentHostContext`、队列、事件、信号、config hook 混在一起。
建议拆分成 `_init_state()` 和 `_install_config_hooks()` 两个私有方法，保持 `__init__` 在 20 行以内。

**`agent_loop.py` — 函数体过长（约 370 行）**
整个 `run_loop` 是一个单体 async generator，包含：abort 检测、runtime snapshot、limit 检查、LLM 调用、tool iteration、python lane 分发。
建议至少把 "tool batch 处理" 和 "limit 检查" 各自提取成私有函数，降低认知负担。

**`agent_loop.py` — 重复的 `is_cancelled` 检查**
`abort_signal.is_cancelled` 在 for 循环内、for 循环外、以及 python lane 分支后共出现 5 次。
可以考虑在每个循环末尾统一 `break`，或封装为守卫函数。

**`handoff.py` — `handoff_session_context` 重复展开 `HandoffResult` 字段**
`handoff_session_context` 函数把 `message_result` 的所有字段逐一展开再重建 `HandoffResult`，但只加了 `session_context`。
建议使用 `dataclasses.replace(message_result, session_context=handed_off_context)` 来避免字段遗漏风险。
