# LLM Provider Layer

`agent_core/llm_provider/` 放模型解析、provider registry、message codec 和各家 API 适配器。

这里负责：

- `provider_types.py` — 模型、provider、compat、usage 等结构定义
- `registry.py` — `models.json` 加载、provider 配置、API key 解析
- `resolver.py` — model uri 解析、默认模型选择、fallback model 生成
- `api_registry.py` — 统一 provider stream 接口与 provider 注册表
- `message_codec.py` — codec 入口与兼容导出
- `codecs/` — provider codec 的正式实现，按 `encoder / decoder / finalize` 对称拆分
- `providers/` — 各厂商适配器：`openai / google / anthropic / minimax / ark / azure`

所有 provider 特殊 replay、signature、raw payload、tool-call 细节都应优先下沉到这一层，而不是堆回 engine loop。

**当前推荐主路径：**
- `encode_messages(...)` — 统一消息到 provider payload
- `decode_chunk(...)` — provider 流式事件到统一 chunk
- `finalize_provider_state(...)` — provider namespaced state 的收尾整理
- `finalize_assistant_message(...)` — assistant message 的最终清理

---

## 审阅建议

### ✅ 做得好的地方
- **`registry.py` 验证逻辑完整**：字段缺失、类型错误均有具体错误消息，且会继续处理剩余 provider，不会整体崩溃。
- **`validation.py` 的类型强制转换（coercion）设计良好**：支持 `"true"/"false"`→bool、`"1"`→int 等宽松类型，方便 LLM 返回不规范参数时兜底。
- **`providers/` 按厂商分文件**，每家 API 的特殊逻辑不会相互污染。

### 🔧 建议改进

**`registry.py` — 行内注释残留**
`_parse_and_merge_providers` 中有两段被注释掉的占位逻辑（`# We inject a placeholder...`），不应提交到正式代码库。建议删除或整理成 TODO。

**`registry.py` — camelCase 字段名**
`Model` 和 `ProviderConfig` 的字段使用 `baseUrl`、`apiKey`、`contextWindow`（camelCase），Python 惯例是 `snake_case`。
建议在加载时做映射转换，内部统一使用 `snake_case`，只在序列化/反序列化边界做格式转换。

**`validation.py` — `_NO_VALUE` 哨兵对象的类型**
`_NO_VALUE = object()` 用作哨兵，但函数返回类型标注为 `bool | object` 等，类型检查器无法推断出哨兵分支。
建议定义一个专用标记类型：`class _Unset: ...` 并使用 `Optional` 或 `Union` 标注，让 mypy 可以正确收窄。

**`codecs/providers.py` 体量较大（约 850 行）**
当前所有 provider 的 encode/decode 逻辑集中在单个文件。建议长期考虑按 provider 拆分，与 `providers/` 目录的结构对齐。
