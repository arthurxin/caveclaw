# Agent Prototype

`Agent_Prototype/` 是产品应用层的原型，基于 `agent_core` 搭建的上层 chat agent。

这里负责：

- `mainchat_agent.py` — 定义 `MainChatAgentConfig` 和抽象基类 `MainChatAgent`
  - `MainChatAgentConfig` 是 `AgentLoopConfig` 协议的具体实现，提供 `model / system_prompt / thinking_level` 等基础配置
  - `MainChatAgent` 是 abstract blueprint，子类通过 `build_system_prompt()` / `build_tools()` / `configure_config()` 定制行为
- `data/` — 存放原型演示所需的数据文件（如 `fake_sales.csv`）

这一层与 `agent_core` 通过接口解耦：`MainChatAgentConfig` 实现了 `convert_to_llm / get_steering_messages / get_followup_messages` 等协议方法，而不直接依赖核心循环的具体实现。

---

## 审阅建议

### ✅ 做得好的地方
- **`MainChatAgent` abstract base class 设计合理**：使用 `@abstractmethod` 强制子类提供 `build_system_prompt()`，`build_tools()` 和 `configure_config()` 可选覆盖，扩展性好。
- **`MainChatAgentConfig` 协议实现完整**：`convert_to_llm / transform_context / get_steering_messages / get_followup_messages` 均有默认实现，子类可以按需覆盖。
- **Demo 函数和私有辅助类以 `_` 前缀命名**（如 `_ExampleMainChatAgent / _demo_stream / _run_demo`），清楚区分了公共 API 和内部示例代码。

### 🔧 建议改进

**目录命名不符合 Python 惯例**
`Agent_Prototype` 使用 PascalCase 加下划线，Python 包/目录惯例是 `snake_case`（全小写）。
建议重命名为 `agent_prototype`，并同步更新 `main.py` 中的 `from Agent_Prototype.mainchat_agent import ...`。

**`mainchat_agent.py` — `_ExampleMainChatAgent` 和 `_run_demo` 应移至 `examples/`**
文件末尾的 `_ExampleMainChatAgent`、`_demo_stream` 和 `_run_demo` 是演示代码，不应混在正式库文件里。
建议移至 `examples/engine_demos/` 或 `examples/provider_demos/` 下的独立文件。

**`MainChatAgentConfig` 缺少类型标注**
类属性 `max_rounds = 20`、`max_consecutive_tool_failures = 5` 未加类型标注（`int`），`__init__` 中的 `temperature / max_tokens` 等也未标注类型。
建议补充完整类型声明，便于 IDE 检查和文档生成。

**`data/` 目录中的 `fake_sales.csv` 职责不明**
`fake_sales.csv` 由 `main.py` 的 `ensure_fake_csv()` 在运行时生成，但 `data/` 目录中已有一份提交版本。
建议明确：是作为静态 fixture 存放还是运行时生成，并在此 README 中说明，避免混淆。
