# Assistant Messages Layer

`agent_core/assistant_messages/` 放统一消息协议和 assistant 流式消息拼装逻辑。

这里负责：

- `types.py`
  block-based message schema、runtime state types、tool/result interfaces、event types。
- `assistant_stream.py`
  把 provider chunk 累积成 `AssistantMessage`，并发出 `message_start / message_update / message_end`。

这层的定位是：

- 向上给 UI / 上层应用提供稳定的 `AgentMessage` / `AssistantMessage`
- 向下给 `core` 和 `llm_provider` 提供统一消息抽象

之后如果接 UI、timeline、inspector panel、rich transcript，也优先放在这一层附近扩展。
