# LLM Provider Layer

`agent_core/llm_provider/` 放模型解析、provider registry、message codec 和各家 API 适配器。

这里负责：

- `provider_types.py`
  模型、provider、compat、usage 等结构定义。
- `registry.py`
  `models.json` 加载、provider 配置、API key 解析。
- `resolver.py`
  model uri 解析、默认模型选择、fallback model 生成。
- `api_registry.py`
  统一 provider stream 接口与 provider 注册表。
- `message_codec.py`
  `AgentMessage <-> provider payload` 的双向 codec。
- `providers/`
  各厂商适配器，如 `OpenAI / Gemini / MiniMax / Ark / Azure Responses`。

所有 provider 的特殊 replay、signature、raw payload、tool-call 细节都应优先下沉到这一层，而不是堆回 engine loop。
