from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from agent_core.llm.api_registry import StreamOptions
from agent_core.llm.message_codec import codec_for_provider
from agent_core.llm.provider_types import Model
from agent_core.llm.registry import ModelRegistry
from agent_core.types import AgentTool, AssistantMessage, Message, ThinkingBlock, ToolCall, ToolResultMessage
from agent_core.assistant_stream import append_assistant_delta
from demo_shared import DOTENV_PATH, MODELS_JSON_PATH, build_provider


class FunctionSchemaTool(AgentTool):
    async def execute(self, tool_call_id: str, params: Dict[str, Any], context, on_update=None, signal=None):
        raise NotImplementedError("Demo tool schema only")


@dataclass
class ToolFunction:
    tool: AgentTool
    func: Callable[[Dict[str, Any]], Any]


def build_demo_tools() -> Dict[str, ToolFunction]:
    weather_tool = FunctionSchemaTool(
        "getWeather",
        "获得指定城市的天气信息",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        },
        "Weather",
    )
    sum_tool = FunctionSchemaTool(
        "getSum",
        "计算数字的和",
        {
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "要相加的数字列表",
                }
            },
            "required": ["numbers"],
        },
        "Sum",
    )

    def get_weather(args: Dict[str, Any]) -> str:
        weather_data = {
            "北京": "晴天, 31°C",
            "上海": "多云, 32°C",
            "广州": "阵雨, 30°C",
        }
        city = str(args["city"])
        return weather_data.get(city, f"{city}: 未知天气")

    def get_sum(args: Dict[str, Any]) -> Any:
        return sum(args["numbers"])

    return {
        "getWeather": ToolFunction(tool=weather_tool, func=get_weather),
        "getSum": ToolFunction(tool=sum_tool, func=get_sum),
    }

def resolve_model(registry: ModelRegistry, provider_name: str, override_model_id: Optional[str]) -> Model:
    if override_model_id:
        model = registry.find(provider_name, override_model_id)
        if model:
            return model

    provider_models = [model for model in registry.get_all() if model.provider == provider_name]
    if not provider_models:
        raise ValueError(f"No models found for provider '{provider_name}'")

    if override_model_id:
        base_model = provider_models[0]
        return Model(
            id=override_model_id,
            provider=provider_name,
            name=override_model_id,
            api=base_model.api,
            baseUrl=base_model.baseUrl,
            reasoning=base_model.reasoning,
            input=list(base_model.input),
            contextWindow=base_model.contextWindow,
            maxTokens=base_model.maxTokens,
            cost=base_model.cost,
            headers=base_model.headers,
            compat=base_model.compat,
        )

    return provider_models[0]


def collect_reasoning(assistant_message: AssistantMessage) -> Optional[str]:
    thinking_blocks = [block.thinking for block in assistant_message.content_blocks if isinstance(block, ThinkingBlock)]
    if not thinking_blocks:
        return None
    return "\n".join(thinking_blocks)


async def call_provider_once(
    provider: Any,
    model: Model,
    messages: List[Message],
    tools: List[AgentTool],
    api_key: Optional[str],
) -> AssistantMessage:
    assistant_message = AssistantMessage(content_blocks=[], raw_content="")
    options = StreamOptions()
    options.tools = tools

    codec = codec_for_provider(provider)
    provider_messages = codec.to_provider_messages(messages, options)
    async for chunk in provider.stream(model, provider_messages, options, api_key=api_key):
        append_assistant_delta(assistant_message, chunk)

    if assistant_message.raw_content == "":
        assistant_message.raw_content = None
    if assistant_message.stop_reason is None:
        assistant_message.stop_reason = "tool_use" if assistant_message.tool_calls else "stop"
    return assistant_message


async def run_provider_demo(provider_name: str, prompt: str) -> None:
    registry = ModelRegistry(MODELS_JSON_PATH)
    provider = build_provider(provider_name)
    override_model_id = os.getenv(f"{provider_name.upper().replace('-', '_')}_MODEL_ID")
    model = resolve_model(registry, provider_name, override_model_id)
    api_key = registry.get_api_key(provider_name)

    tool_functions = build_demo_tools()
    tool_schemas = [item.tool for item in tool_functions.values()]

    system_prompt = "你是一个智能助手，可以调用工具获取天气信息和计算数字。"
    messages: List[Message] = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=prompt),
    ]

    print(f"\n=== Provider: {provider_name} / Model: {model.id} ===")
    print(f"用户问题: {prompt}")

    round_num = 1
    while True:
        assistant_message = await call_provider_once(provider, model, messages, tool_schemas, api_key)
        messages.append(assistant_message)

        reasoning = collect_reasoning(assistant_message)
        print(f"\n[第 {round_num} 轮推理]")
        print(reasoning if reasoning else "(无 reasoning)")
        print(f"\n[回复内容]\n{assistant_message.content}")
        print(f"\n[raw_content]\n{assistant_message.raw_content}")
        print(f"\n[工具调用]\n{assistant_message.tool_calls}")

        if not assistant_message.tool_calls:
            print("\n最终回复:")
            print(assistant_message.content)
            return

        for tool_call in assistant_message.tool_calls:
            tool_binding = tool_functions.get(tool_call.name)
            if not tool_binding:
                raise ValueError(f"未知工具: {tool_call.name}")
            result = tool_binding.func(tool_call.arguments)
            print(f"  > 调用 {tool_call.name}({json.dumps(tool_call.arguments, ensure_ascii=False)}) => {result}")
            messages.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=str(result),
                )
            )

        round_num += 1


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider tool-calling smoke demo.")
    parser.add_argument(
        "--providers",
        default="minimax,google,openai,anthropic,volcengine,azure",
        help="Comma-separated provider names to test.",
    )
    parser.add_argument(
        "--prompt",
        default="北京和上海的天气怎么样？把这两个城市的温度加起来",
        help="Prompt to send.",
    )
    args = parser.parse_args()

    load_dotenv(DOTENV_PATH)

    for provider_name in [item.strip() for item in args.providers.split(",") if item.strip()]:
        try:
            await run_provider_demo(provider_name, args.prompt)
        except Exception as error:
            print(f"\n=== Provider: {provider_name} ===")
            print(f"运行失败: {error}")


if __name__ == "__main__":
    asyncio.run(main())
