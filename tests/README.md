# Tests

`tests/` 放所有单元测试和集成测试。使用标准 pytest 框架运行。

**运行测试：**
```bash
uv run pytest tests/
```

## 测试文件索引

| 文件 | 覆盖范围 |
|---|---|
| `test_agent.py` | `Agent` 高层控制面 |
| `test_agent_loop.py` | `run_loop` 核心循环 |
| `test_handoff.py` | handoff 消息转换 |
| `test_handoff_matrix.py` | handoff 场景矩阵测试 |
| `test_cross_provider_handoff.py` | 跨 provider handoff |
| `test_message_codec.py` | provider 消息编解码 |
| `test_transform_messages.py` | 消息转换规则 |
| `test_compaction.py` | 上下文压缩 |
| `test_compat.py` | provider compat 检查 |
| `test_llm_provider_api.py` | provider API 接口 |
| `test_model_registry.py` | `ModelRegistry` 加载与验证 |
| `test_model_resolver.py` | `ModelResolver` 解析逻辑 |
| `test_validation.py` | 工具参数校验 |
| `test_inspector.py` | runtime inspector |
| `test_python_program_execution.py` | python lane 执行 |
| `test_openai_provider.py` | OpenAI 适配器 |
| `test_google_provider.py` | Google/Gemini 适配器 |
| `test_anthropic_provider.py` | Anthropic 适配器 |
| `test_azure_provider.py` | Azure 适配器 |
| `test_minimax_provider.py` | MiniMax 适配器 |
| `test_main_dataframe_flow.py` | dataframe 端到端流程 |
| `test_mainchat_agent_prototype.py` | `MainChatAgent` 原型 |

---

## 审阅建议

### ✅ 做得好的地方
- **覆盖面广**：测试文件对应项目每一个主要模块，层次清晰，单元测试和集成测试混合放置但命名规范。
- **按模块命名**：`test_<module>.py` 命名规范，容易找到某个模块对应的测试。

### 🔧 建议改进

**缺少 `conftest.py`**
当前没有 `conftest.py`，所有测试中用到的 fixture（如 mock provider、fake model、测试用 AgentLoopConfig）均在各文件内重复定义。
建议提取共用 fixture 至 `conftest.py`，减少重复、提高一致性。

**`test_agent.py` 和 `test_agent_loop.py` 体量较大**
`test_agent_loop.py` 约 1070 行，`test_agent.py` 约 670 行，建议按场景拆分为更小的文件（例如 `test_agent_abort.py`、`test_agent_handoff.py`），提升可维护性。

**Provider 测试缺少统一入口说明**
`test_openai_provider.py` / `test_google_provider.py` 等需要真实 API Key 才能运行，但没有说明如何跳过或 mock。
建议统一使用 `@pytest.mark.live` 等标记区分需要网络的测试，并在此 README 中说明如何仅运行本地测试。
