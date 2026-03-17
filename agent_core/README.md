# Agent Core

`agent_core/` 现在按三层来组织：

- `core/`
  负责运行时内核、loop、tool execution、runtime projection、inspector、compaction。
- `assistant_messages/`
  负责统一的消息协议、`AssistantMessage` 组装、事件流消息累积。
- `llm_provider/`
  负责模型注册、provider codec、provider 适配器，以及各家 API 的特殊 replay 逻辑。

设计边界是：

- `core` 可以依赖 `assistant_messages` 和 `llm_provider`
- `assistant_messages` 不依赖 `core`
- `llm_provider` 不依赖 `core`

这样可以保持消息协议、LLM 适配、执行引擎三者分层清晰。

更具体的职责请看每个子目录下自己的 `README.md`。
