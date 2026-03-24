# Docs

`docs/` 放设计文档、架构说明和参考资料。

## 当前内容

- `archive/` — 归档的早期设计文档（当前为空）

---

## 审阅建议

### 🔧 建议改进

**`archive/` 目录为空**
空目录在 git 中不会被追踪。如果该目录用于归档历史文档，应放入实际内容，或删除该目录，等有内容时再创建。

**`docs/` 目前没有实际文档**
项目架构知识目前分散在各模块的 `README.md` 中，`docs/` 实际未被利用。
建议将以下文档逐步补充到 `docs/`:
- `architecture.md` — 整体架构图和三层依赖关系
- `getting_started.md` — 从零开始运行项目的步骤
- `provider_setup.md` — 各 provider 的 API Key 和 `models.json` 配置方法
- `events_reference.md` — `AgentEvent` 类型和事件字段说明
