# Agent Core

`agent_core/` 按三层组织：

- `core/` — 运行时内核、loop、tool execution、runtime projection、inspector、compaction
- `assistant_messages/` — 统一消息协议、`AssistantMessage` 组装、事件流累积
- `llm_provider/` — 模型注册、provider codec、各家 API 适配器

**依赖方向（严格遵守）：**
- `core` 可以依赖 `assistant_messages` 和 `llm_provider`
- `assistant_messages` 不依赖 `core`
- `llm_provider` 不依赖 `core`

---

## 审阅建议

### ✅ 做得好的地方
- **三层分离清晰**，`core / assistant_messages / llm_provider` 依赖方向一致，没有反向依赖。
- **`__init__.py` 显式 `__all__`** 列表完整，公开接口边界明确。
- **类型标注覆盖率高**，主要 data class 和函数签名均有类型。

### 🔧 建议改进
- **`__init__.py` 排序混乱**：`__all__` 中的名称排列和 `from ... import ...` 的顺序不一致，建议按字母序统一排列，方便查找。
- **`SyntheticToolResultPolicy` 出现在 `__all__` 末尾**，但在 `from ... import` 中导入位置靠前，保持顺序一致。
- **顶层 `__pycache__/` 目录**应加入 `.gitignore`（已加入的请忽略），避免提交。
