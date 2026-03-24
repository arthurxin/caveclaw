# Assistant Messages Layer

`agent_core/assistant_messages/` 放统一消息协议和 assistant 流式消息拼装逻辑。

这里负责：

- `types.py` — block-based message schema、runtime state types、tool/result interfaces、event types
- `content_blocks.py` — `TextBlock / ThinkingBlock / ToolCallBlock / ToolResultBlock / ImageBlock / RuntimeRefBlock` 等内容块定义
- `messages.py` — `Message / AssistantMessage / ToolResultMessage / CustomMessage` 等消息类
- `state.py` — `AgentState / AgentEvent / AgentLoopConfig / AssistantDelta`
- `runtime.py` — `RuntimeState / RuntimeVariable / RuntimeDeltaOp / AgentContext / ToolRuntimeSelection`
- `tools.py` — `AgentTool / AgentToolResult / CancellationSignal / StateReducer`
- `assistant_stream.py` — 把 provider chunk 累积成 `AssistantMessage`，发出 `message_start / message_update / message_end`
- `transform_messages.py` — 跨 provider/model 的消息转换规则

**定位：**
- 向上给 UI / 上层应用提供稳定的 `AgentMessage` / `AssistantMessage`
- 向下给 `core` 和 `llm_provider` 提供统一消息抽象
- 如果接 UI、timeline、inspector panel、rich transcript，也优先在这一层扩展

---

## 审阅建议

### ✅ 做得好的地方
- **`messages.py` 的 `content / content_blocks` 双向同步**：通过 `@property` 和 `@content.setter` 无缝对外提供文本 API，同时内部保留结构化 blocks。
- **`types.py` 作为统一导出入口**，下层模块（`content_blocks / messages / runtime / state / tools`）分文件组织清晰，不混杂。
- **`transform_messages.py` 集中处理跨 provider 适配**，不渗透到 engine loop。

### 🔧 建议改进

**`messages.py` — `Message.__init__` 绕过 dataclass 默认构造**
`@dataclass(kw_only=True, init=False)` 加手写 `__init__` 的组合是为了支持 `content` / `content_blocks` 互斥入参。
但 `dataclass` 会生成 `__repr__` / `__eq__` 等依赖字段声明，而手写 `__init__` 不调用 dataclass 生成的 `__init__`，容易在子类继承时出现意外。
建议用 `@dataclass` + `__post_init__` 替代，或直接去掉 `@dataclass` 装饰器，改为普通类，使意图更清晰。

**`transform_messages.py` 体量过大（约 1068 行）**
这个文件承载了几乎所有的消息转换规则，建议按转换关注点拆分子模块，例如：
- `_rewind.py` — 消息回退逻辑
- `_provider_state.py` — provider state 的 drop / preserve 规则
- `_synthetic_tool_results.py` — 合成 tool result 的插入逻辑

**`state.py` — `AgentLoopConfig` 字段动态修改**
`agent.py` 通过 `setattr(config, "xxx", ...)` 动态给 `AgentLoopConfig` 注入字段（如 `get_steering_messages`、`abort_signal`）。
建议在 `AgentLoopConfig` 中显式声明这些可选字段，避免依赖动态属性，提高类型安全性。
