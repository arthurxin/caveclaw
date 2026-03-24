# Examples

这个目录只放示例和实验脚本，不承载正式库代码。

- `engine_demos/` — 演示 `agent_core` 的事件流、熔断和状态管理
- `provider_demos/` — 演示多厂商 provider、模型解析、工具调用和流式输出
- `legacy/` — 早期基于 MiniMax 的原型实现，保留作设计参考

---

## 审阅建议

### ✅ 做得好的地方
- **按用途分子目录**：`engine_demos` 和 `provider_demos` 职责分明，`legacy` 单独隔离不污染主流程。

### 🔧 建议改进

**`legacy/` — 两个文件缺乏说明**
`agent_loop.py` 和 `minimax_calling.py` 是旧版代码，但没有任何注释说明它们和当前实现的区别，或哪些部分已经被替代。
建议在 `legacy/README.md` 中补充说明，或在文件顶部添加 `# LEGACY: ...` 注释，明确废弃原因。

**`provider_demos/` — 缺少 README**
`provider_demos/` 下的 5 个 demo 文件没有 README 说明各自演示什么场景、需要什么配置。建议补充一个 `README.md`。

**`engine_demos/agent_scenarios_demo.py` — 作为入口不够显眼**
这是演示 agent 完整场景的核心文件，但需要阅读代码才能知道如何运行。建议在 README 中补充一行快速运行命令示例。
