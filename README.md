# CaveClaw

CaveClaw 是一个面向工具调用型 Agent 的 Python 项目，可以把它理解为对 `pi-mono` / `pi-agent-core` 思路的一次 Python 化、事件流化改写。

现在仓库里的主线已经比较清楚：

- `agent_core/` 是正式核心库。
- `examples/` 是可运行的演示与历史原型。
- `docs/archive/` 是保留的旧设计说明。
- `tests/` 只放最小回归测试。

## 现在写得怎么样

整体评价：**方向是对的，核心想法有辨识度，但之前仓库把“核心库、历史原型、演示脚本、文档草稿”混在了一起，所以读起来比实际复杂。**

我认为目前最有价值的部分有三点：

1. `agent_core/core/agent_loop.py` 的事件流设计是成立的。
   它把 Agent 过程拆成了 `agent_start / turn_start / tool_execution_* / turn_end / agent_end` 这类结构化事件，这比单纯返回字符串更适合做 UI、日志和调试。
2. `state_delta + inspector` 的组合是个很好的方向。
   你不是把巨大运行态直接塞回模型，而是通过 `ToolResult.state_delta` 更新共享状态，再用 `PythonRuntimeInspector` 做摘要，这个边界设计是清楚的。
3. 多 provider 的统一抽象已经有雏形。
   `ModelRegistry / ModelResolver / ApiProvider` 这一层虽然还在早期，但方向比“把 SDK 调用散落在业务逻辑里”健康很多。

同时，之前也有几个明显问题：

- `src/` 的旧原型和 `agent_core/` 并存，主线不够明确。
- `tests/` 里混入了大量 demo 脚本，不利于回归验证。
- 根目录 `README.md` 为空，外部读者无法快速理解项目现状。
- 统一 provider 路径下，工具定义没有传给 provider，这会影响真实 tool calling。

这些问题本轮已经顺手整理了一遍。

## 本次整理后的目录

```text
caveclaw/
├─ agent_core/
│  ├─ core/                 # 执行引擎、loop、runtime、compaction、inspector
│  ├─ assistant_messages/   # AgentMessage / AssistantMessage / block schema / message streaming
│  ├─ llm_provider/         # ModelRegistry / Resolver / codec / providers
│  └─ README.md
├─ docs/archive/            # 旧版设计文档
├─ examples/
│  ├─ engine_demos/         # 事件流与 Agent 行为演示
│  ├─ provider_demos/       # provider / resolver / tool calling 演示
│  └─ legacy/               # 早期 MiniMax 原型
├─ scripts/                 # 辅助脚本
├─ tests/                   # 最小回归测试
├─ models.json              # 模型与 provider 配置
└─ todo.md                  # 当前路线图
```

## 设计判断

如果继续往下做，我建议你把 CaveClaw 明确收敛成下面这条主线：

- `agent_core/core` 负责执行引擎、runtime 驱动 loop、状态提交和上下文压缩。
- `agent_core/assistant_messages` 负责统一消息协议和面向 transcript / UI 的 rich message 层。
- `agent_core/llm_provider` 负责模型解析、provider 适配、流式输出归一化。
- 上层应用再去决定 prompt、tools、memory persistence、UI 展现。

### 现在的内核方向

当前主线已经不再是“字符串消息 + 工具回文本”的简单 agent loop，而是在往一个 **runtime 驱动内核** 收敛：

- `RuntimeState` 保存工具真正改动过的世界状态。
- `AgentMessage` 是统一的 rich block 协议，用来承载 LLM 可见信息。
- provider 只是 `AgentMessage` 之下的实现细节，`minimax_local`、`gemini` 这类 replay / signature 逻辑放在 provider codec 内处理。

这条线的核心原则是：

**runtime 是一等状态，message 是 runtime 的投影。**

### Runtime 驱动 loop 的当前节奏

现在的 `agent_loop` 已经按下面的方式运行：

1. loop 开始时注入当前 runtime snapshot。
2. 调模型，判断是否需要工具。
3. 如果进入一批工具调用：
   工具只返回 `runtime_ops`，由 loop 在这一批结束后统一 commit。
4. commit 后重新生成 runtime snapshot，并追加一条短 worklog。
5. 将新的 snapshot 和 worklog 作为 `AgentMessage` 放回上下文，再看模型是否还要继续当前工具链。
6. 当当前工具链结束后，带着新的 snapshot 和 worklog 进入下一次外层审查。

这意味着主循环已经从原来的“文本式循环”升级成了“runtime 驱动循环”。

### Compaction 的位置

这里的 compaction 不是 session 层的大而全记忆系统，而是一个每次调用 provider 前都要经过的步骤：

- 输入：`AgentMessage`
- 输出：发给具体 `llm_provider` 的 message payload

基础职责主要有两类：

- 去掉不该给 LLM 的 UI-only / log-only 富文本
- 控制 runtime snapshot 对 LLM 暴露的粒度，优先传变量 metadata 和紧凑摘要，而不是完整 UI 展示信息

现在已经有一个基础版独立模块，挂在 `transform_context -> compaction -> convert_to_llm` 这条链路里；后续再继续扩到 token budget 和 provider 特化规则。

### Provider Codec 的方向

现在已经开始把 `raw_content / provider_state` 的解释逻辑下沉到 provider codec：

- `AgentMessage` 继续保持统一。
- `provider_state` 用 namespace 保存厂商特定 replay 信息。
- provider codec 决定怎么把 `AgentMessage` 转成各家真正要吃的 payload。

这让像 MiniMax 的 `<think>` 原文回放、Gemini 的 `thoughtSignature` / parts replay 都可以留在 provider 层，而不用污染统一消息协议。

换句话说，**核心层尽量“少做业务，多做边界”**。这一点你现在已经走在正确方向上了，后面最该防的是继续把临时业务逻辑塞回核心。

## 快速开始

1. 准备 Python 3.12。
2. 安装依赖。
3. 配置 `.env` 里的 API Key。
4. 根据需要编辑 `models.json`。
5. 先看 `examples/` 再接入自己的上层应用。

## 默认模型

目前框架层的默认初始模型是：

- `openai/gpt-5.4`

这是 `ModelResolver.find_initial_model()` 的默认值，也是当前 `models.json` 中与 resolver 对齐的主默认模型。

另外，示例脚本会做一点更贴近本地环境的选择：

- `provider_stream_demo.py` / `agent_messages_demo.py`
  如果 `.env` 里设置了 `CAVECLAW_DEFAULT_PROVIDER` 和 `CAVECLAW_DEFAULT_MODEL`，会优先按这两个值选择。
- 如果没有显式设置上述参数，但检测到 `AZURE_API_KEY`
  示例会优先默认使用 `azure/gpt-5.4`。
- 其他 resolver / registry 主路径
  仍以 `openai/gpt-5.4` 作为框架默认入口。

示例入口：

- `examples/engine_demos/agent_scenarios_demo.py`
- `examples/provider_demos/provider_stream_demo.py`
- `examples/provider_demos/agent_messages_demo.py`
- `examples/provider_demos/provider_tool_calling_demo.py`

## Azure Provider

仓库现在已经内置了一个基于 Azure OpenAI Responses API 的 `azure` provider：

- provider 名称：`azure`
- api 类型：`azure-responses`
- 默认模型：`gpt-5.4`
- 默认 endpoint：`https://xinhongyu-resource.openai.azure.com/openai/v1/responses`

它当前支持的能力包括：

- `instructions + input` 形式的首轮请求构造
- reasoning effort 映射
- 原生 Responses API function calling
- `previous_response_id` 续轮
- `function_call_output` 工具结果回传

### Azure 环境变量

最少需要：

- `AZURE_API_KEY`
- `CAVECLAW_DEFAULT_PROVIDER`
  如果你希望在本地环境中默认走某个 provider，例如 `azure`。

可选：

- `CAVECLAW_DEFAULT_MODEL`
  与 `CAVECLAW_DEFAULT_PROVIDER` 配合使用，例如 `gpt-5.4`。
- `AZURE_BASE_URL`
  如果不填，则使用 `models.json` 中的默认 endpoint。
- `AZURE_AUTH_MODE`
  可选 `bearer` 或 `api-key`。
  默认是 `bearer`，因为你当前给的是这种认证方式。
- `AZURE_MODEL_ID`
  用于覆盖默认 `gpt-5.4`。

### Azure 测试方式

真实 provider tool calling 烟雾测试：

```bash
PYTHONPATH=/Users/arthurxing/Desktop/project/github/caveclaw \
.venv/bin/python examples/provider_demos/provider_tool_calling_demo.py --providers azure
```

普通流式输出测试：

```bash
.venv/bin/python examples/provider_demos/provider_stream_demo.py azure/gpt-5.4
```

基于 `AgentMessage` 的 provider 测试：

```bash
.venv/bin/python examples/provider_demos/agent_messages_demo.py
```

目前 Azure provider 已经通过本地单测，以及真实 Azure endpoint 的多轮 tool calling smoke test。

## 代码整理补充

这轮除了 provider 和 runtime 的内核改动，也顺手做了两类结构整理：

- `examples/provider_demos/demo_shared.py`
  抽出了 demo 共享的 provider 注册、provider 构造、项目路径常量，减少多个示例脚本里的重复样板代码。
- `agent_core/llm_provider/providers/`
  现在明确承载 provider 适配器；`AzureProvider` 已放在这一层，与 `GoogleProvider / MiniMaxProvider / ArkProvider` 并列。
- `agent_core/core/`、`agent_core/assistant_messages/`、`agent_core/llm_provider/`
  已正式拆层，并各自带有 README，用来约束执行内核、消息协议、LLM 适配三者的边界。

## 当前建议的开发顺序

1. 先把 `agent_core` 这条主线继续收紧。
2. 给 `run_loop`、`resolver`、`inspector` 补稳定单测。
3. 再做 prompt builder、skills、memory persistence 这类增强层。

## 相关文档

- 模块说明见 `agent_core/README.md`
- 历史设计稿见 `docs/archive/`
- 近期任务见 `todo.md`
