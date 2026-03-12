"""
测试脚本：使用 AgentMessage (types.py Message 格式) 测试各 Provider 的流式输出。

运行方式:
    .venv/bin/python tests/test_agent_message.py
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core.types import Message, AssistantMessage, AgentMessage
from agent_core.llm.registry import ModelRegistry
from agent_core.llm.resolver import ModelResolver
from agent_core.llm.api_registry import api_provider_registry, StreamOptions
from agent_core.llm.providers import (
    OpenAiProvider, AnthropicProvider, GoogleProvider,
    MiniMaxProvider, ArkProvider
)

# 注册所有 Provider
api_provider_registry.register(OpenAiProvider())
api_provider_registry.register(AnthropicProvider())
api_provider_registry.register(GoogleProvider())
api_provider_registry.register(MiniMaxProvider())
api_provider_registry.register(ArkProvider())


async def test_with_agent_messages(model_uri: str):
    """用 AgentMessage 格式测试一个 Provider 的流式输出。"""
    print(f"\n{'='*50}")
    print(f"  Testing: {model_uri}")
    print(f"{'='*50}")

    # 1. 解析模型
    registry = ModelRegistry()
    resolver = ModelResolver(registry)
    model, thinking_level = resolver.resolve(model_uri)
    if not model:
        print(f"  ❌ 无法解析模型: {model_uri}")
        return

    print(f"  Model:   {model.provider}/{model.id}")
    print(f"  API:     {model.api}")
    print(f"  Thinking: {thinking_level}")

    # 2. 获取 API Key
    api_key = registry.get_api_key(model.provider)
    if not api_key:
        print(f"  ❌ 未找到 API Key (Provider: {model.provider})")
        return

    # 3. 构建 AgentMessage 对话历史（使用 Message 对象，不是裸 dict）
    conversation: list[AgentMessage] = [
        Message(role="user", content="你好！请用中文做一段极简自我介绍（不超过两句话），然后告诉我今天你服务的提供商是谁。"),
    ]

    # 4. 转为 LLM 格式 (to_dict)
    llm_messages = [m.to_dict() for m in conversation]

    # 5. 获取 Provider 并流式输出
    provider = api_provider_registry.get(model.api)
    if not provider:
        print(f"  ❌ 未注册 Provider: {model.api}")
        return

    options = StreamOptions()
    options.thinking_level = thinking_level

    print(f"\n  Response:\n  {'-'*40}")
    full_content = ""
    reasoning_content = ""

    try:
        async for chunk in provider.stream(
            model=model,
            messages=llm_messages,
            options=options,
            api_key=api_key,
        ):
            if "reasoning" in chunk:
                reasoning_content += chunk["reasoning"]
            if "content" in chunk:
                full_content += chunk["content"]
                print(chunk["content"], end="", flush=True)
            if "tool_calls" in chunk:
                print(f"\n  [Tool Calls]: {chunk['tool_calls']}")

        print(f"\n  {'-'*40}")
        if reasoning_content:
            preview = reasoning_content[:80] + "..." if len(reasoning_content) > 80 else reasoning_content
            print(f"  [Reasoning Preview]: {preview}")

        # 6. 演示构建 AssistantMessage 对象（模拟 agent_loop 内部行为）
        assistant_msg = AssistantMessage(content=full_content)
        conversation.append(assistant_msg)
        print(f"  ✅ AssistantMessage 构建成功 (len={len(full_content)} chars)")
        print(f"  对话长度: {len(conversation)} messages")

    except Exception as e:
        print(f"\n  ❌ 错误: {str(e)}")


async def main():
    load_dotenv()

    targets = [
        "google/gemini-pro-latest",
        "minimax/minimax",
        # ARK 需要真实 endpoint ID，跳过占位符
    ]

    # 如果 .env 里有 ARK_API_KEY，就测一下
    if os.environ.get("ARK_API_KEY"):
        targets.append("volcengine/ep-xxxxxxxx-xxxxx")  # 替换为实际 endpoint ID

    for model_uri in targets:
        await test_with_agent_messages(model_uri)

    print(f"\n{'='*50}")
    print("  All tests complete.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
