# Core Layer

`agent_core/core/` 放 CaveClaw 的执行引擎内核。

这里负责：

- `agent.py`
  `Agent` 高层控制面，维护 steering / follow-up / continue / abort / wait_for_idle。
- `agent_loop.py`
  runtime 驱动主循环，负责 turn、tool batch、runtime snapshot、worklog、结束条件。
- `tool_execution.py`
  工具执行、partial update、取消信号、staged runtime ops 收集。
- `runtime_projection.py`
  `RuntimeState -> AgentMessage` 的投影逻辑，包括 snapshot、worklog、runtime commit。
- `compaction.py`
  发送给 LLM 前的上下文裁剪与消息压缩。
- `inspector.py`
  runtime 变量摘要、Reducer、环境状态抓取。

这层不直接处理厂商 API 协议，也不承载 UI 逻辑。
