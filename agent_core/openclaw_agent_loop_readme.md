# CaveClaw Agent Core

`caveclaw.agent_core` 是一个轻量级、完全受控且高度容纳自定义 UI 的大型语言模型 (LLM) 推理闭环引擎。它是基于 `asyncio` 的 TypeScript `pi-agent-core` 的原生 Python 复刻版。

## 🎯 核心设计哲学

原生的 LLM 接口（如 OpenAI, Anthropic）过于死板，只允许 `role: user/assistant/system`。如果你想在会话历史中持久化地保留“系统通知”、“正在加载的占位符卡片”甚至“报错黄条”，强塞给 LLM 会导致 Token 浪费和幻觉报错。

`agent_core` 解决了这个问题：
1. **统一时间线 (`AgentMessage`)**：在应用层，你可以往 `agent.state.messages` 里随意放入标准的 LLM 消息，或者你自定义的任意 UI 消息（如 `NotificationMessage`）。
2. **清洗拦截 (`convert_to_llm`)**：在将上下文发往 LLM 产生下一次推理前，框架会触发该钩子。你可以自由地“过滤掉 UI 消息”、“降维压缩长文本”或“转化复杂的卡片为简单可读的系统报告”。
3. **原生工具支持 (`AgentTool`)**：内置对 Function Calling 的闭环支持。模型输出的工具调用会自动分发到相应的 Python 异步函数上，并且允许你在漫长的执行中抛出动态进度（`on_update`）。
4. **动态干预 (`Steering`)**：模型或工具在跑的时候，用户突然又发了一句话？直接调用 `agent.steer()`，引擎会立刻中止排队中的其他动作，并把新指令加入上下文立刻开启新的推理轮回！

---

## 🛠 快速起步

### 1. 业务定义：定制属于你的 UI 消息与工具

假设你想让 AI 能查天气，并在发请求的中途展示一个加载 UI。

```python
from dataclasses import dataclass
from caveclaw.agent_core import CustomMessage, AgentTool, AgentToolResult

@dataclass
class LoadingCard(CustomMessage):
    """一个不需要给大模型看的 UI 加载卡片"""
    custom_type: str = "loading_ui"
    text: str = ""

class WeatherTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            label="获取实时天气",
            description="获取给定城市的天气信息",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}}
        )

    async def execute(self, tool_call_id: str, params: dict, on_update=None) -> AgentToolResult:
        # 在这里执行你真正的业务逻辑
        import asyncio
        await asyncio.sleep(1) # 模拟网络请求
        city = params.get("city", "未知城市")
        return AgentToolResult(
            content=f"{city}今天晴朗，25度。", 
            details={"raw_data": {"temp": 25, "condition": "sunny"}}
        )
```

### 2. 配置引擎大脑：实现 `AgentLoopConfig`

你需要告诉引擎，如何把大杂烩的消息净化成干净的结构，以及是否有人工干预。

```python
from caveclaw.agent_core import AgentLoopConfig, AgentMessage, Message

class MyConfig(AgentLoopConfig):
    async def convert_to_llm(self, messages: list[AgentMessage]) -> list[Message]:
        clean_msgs = []
        for msg in messages:
            # 标准消息直接放行
            if isinstance(msg, Message):
                clean_msgs.append(msg)
            # 遇到 UI 加载卡片，大模型不需要知道，直接滤除！
            elif isinstance(msg, LoadingCard):
                continue
        return clean_msgs
        
    async def transform_context(self, messages):
        # 预留的上下文压缩钩子，比如在这里可以统一切除过老的聊天记录
        return messages
```

### 3. 主程序：创建 Agent 并启动流式事件引擎

`agent.prompt()` 是一个非常灵活的异步生成器，它会在执行的不同阶段吐出 `AgentEvent` 事件，非常适合用来挂接前端 WebSocket。

```python
import asyncio
from caveclaw.agent_core import Agent, UserMessage

# 假设这是一个对齐 OpenAI 流式响应格式的包裹函数 (需自行对接真实的 LLM)
async def mock_stream_llm(messages):
    yield {"content": "今天"}
    yield {"content": "天气不错"}
    
async def main():
    agent = Agent(config=MyConfig(), tools=[WeatherTool()])
    agent.set_system_prompt("你是一个得力的气象助手。")
    
    # 用户提问！
    user_msg = UserMessage(content="北京今天热吗？")
    
    # 模拟前端收听运行状态事件
    async for event in agent.prompt(user_msg, mock_stream_llm):
        if event.type == "agent_start":
            print("▶️ 推理启动...")
        elif event.type == "turn_start":
            print("🔄 新的一轮推理...")
        elif event.type == "turn_end":
            msg = event.data["message"]
            print(f"✅ 回复完成: {msg.content}")
            if "tool_results" in event.data and event.data["tool_results"]:
                print(f"🔧 执行了工具，并回收结果！")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧩 核心 API 纵览

### `Agent` (外观模式核心)
- `agent.steer(message)`: **神技**。当 Agent 正在执行一长串任务时，你可以强行塞入一条引导消息，它会在当前工具完成后立刻跳过剩余任务，直接带上你的引导发起下一轮 LLM 对话。
- `agent.follow_up(message)`: 追加一条消息，但不同于 steer，它一定要等 AI 自己把话全说完、所有附带工具全部执行完且主动停下后，才会触发处理。
- `agent.replace_messages(messages)`: 覆盖完整的历史记录，适合于“恢复保存的会话草稿”。

### `AgentEvent` 生命周期
从 `agent.prompt(...)` 吐出的事件极速响应 UI：
- `agent_start` / `agent_end`
- `turn_start` / `turn_end` (夹带着这段 turn 生成出来的最新一句话 `message` 及其触发的 `tool_results`)
