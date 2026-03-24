# Scripts

`scripts/` 放独立的工具脚本，供开发和调试使用，不是库代码的一部分。

## 当前文件

- `list_gemini_models.py` — 列出当前 Gemini API 下可用的模型列表

**运行方式：**
```bash
uv run python scripts/list_gemini_models.py
```

---

## 审阅建议

### ✅ 做得好的地方
- 工具脚本独立存放，不混入库代码或测试目录。

### 🔧 建议改进

**脚本数量少，但目录已有扩展价值**
当前只有一个脚本，但随着添加更多 provider，类似的 `list_openai_models.py`、`list_anthropic_models.py` 等都适合放这里。
建议在此 README 中维护一个快速索引，方便开发者查找可用脚本。

**`list_gemini_models.py` 缺少错误处理**
如果 API Key 缺失或网络不通，脚本会直接抛出异常，缺少友好的错误提示。
建议添加 `try/except` 和清晰的帮助信息，例如 `请设置 GOOGLE_API_KEY 环境变量`。
