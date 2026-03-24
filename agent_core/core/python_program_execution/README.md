# Python Program Execution Lane

`agent_core/core/python_program_execution/` 放 `"""python"""` / fenced python block 驱动的独立执行通道。

这条 lane 与 `tool_execution.py` 的 native tool calling lane 并列存在，但职责不同：

- **native tool lane** — 处理 provider 原生 `tool_calls`
- **python program lane** — 处理 assistant 产出的 python 代码块，并交给独立执行器运行

这里负责：

- `types.py` — python program block、执行请求、执行结果等结构定义
- `parser.py` — 从 assistant 文本里提取 python block，支持 ` ```python``` ` 和 `"""python`
- `executor.py` — 独立执行器，支持 `ipython` 与 `python` 两种 backend
- `lane.py` — loop 侧的独立分派、事件、执行结果消息与 worklog 生成；提供 python lane 开关与 controller
- `bridge.py` — 把 python 执行后 namespace 中的可桥接变量自动同步到 `RuntimeState`；支持自定义 bridge

**注意：**
- 这条 lane 已接进 `agent_loop.py`，不会替代 native tool lane
- loop 显式分派：优先走 provider-native `tool_calls`，否则检查 `python program`

**配置方式：**
```python
config.python_program_execution = False          # 关闭 python lane
config.python_program_execution = True           # 使用默认 controller
config.python_program_execution = YourController()  # 自定义 controller
config.python_runtime_bridge = False             # 关闭 namespace → runtime 同步
config.python_runtime_bridge = YourBridge()     # 自定义 bridge
```

---

## 审阅建议

### ✅ 做得好的地方
- **单一职责执行通道**：python lane 完全独立于 native tool lane，不复用也不干扰，结构清晰。
- **`bridge.py` 的自动同步设计**：默认 bridge 能识别 DataFrame、标量、dict/list，并过滤模块和函数，属于合理的保守策略。
- **`executor.py` 支持双 backend**：`ipython` 和 `python` 均可配置，不强绑依赖。

### 🔧 建议改进

**`parser.py` 逻辑简单但缺少文档说明**
当前 `parser.py` 仅约 50 行，实现了两种 block 格式的提取，但没有注释说明支持的格式边界（例如：多个 block 时只取第一个？支持嵌套 block 吗？）。建议补充 docstring 说明前提假设。
此处应该改为仅获取第一个python_block, 再阻断后续的llm回复, 不把后续的llm回复加入message. 在执行完python_block后, 直接把后续的python runtime执行结果加入message, 回复给模型看有没有下一个block. 我们不做嵌套block.

**`lane.py` — `execute_python_program_lane` 参数过多**
函数接受 `assistant_message / python_block / agent_context / config / turn_event_id` 共 5 个参数，且均无默认值。建议封装成一个 `PythonLaneRequest` dataclass 作为单参数传入，降低调用侧认知负担。

**`bridge.py` — 异常处理过于宽泛**
bridge 内部有多处 `except Exception: pass` 或静默跳过，这会让调试变量桥接问题变得困难。建议至少记录警告日志，或将错误附加到执行结果对象中。
