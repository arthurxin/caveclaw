"""
Volcengine ARK Provider adapter.
火山引擎方舟大模型 API 适配器。

使用官方 volcenginesdkarkruntime SDK（而非 OpenAI SDK），
支持 doubao 系列模型，model 参数使用「模型名」或「endpoint_id」均可。

安装依赖:
    uv add 'volcengine-python-sdk[ark]'

.env 配置:
    ARK_API_KEY=your-ark-api-key
"""
import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions

ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class ArkProvider:
    """
    Implements the ApiProvider protocol for Volcengine ARK models.
    Uses the official volcenginesdkarkruntime (Ark client).

    base_url resolution priority:
      1. model.baseUrl (from models.json, if set)
      2. ARK_BASE_URL env var
      3. default: https://ark.cn-beijing.volces.com/api/v3
    """
    api = "ark"

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            from volcenginesdkarkruntime import AsyncArk
        except ImportError:
            raise ImportError(
                "volcenginesdkarkruntime package is required: "
                "uv add 'volcengine-python-sdk[ark]'"
            )

        key = api_key or os.environ.get("ARK_API_KEY")
        if not key:
            raise ValueError("No ARK API key found. Set ARK_API_KEY in .env")

        base_url = (
            model.baseUrl
            or os.environ.get("ARK_BASE_URL")
            or ARK_DEFAULT_BASE_URL
        )

        client = AsyncArk(api_key=key, base_url=base_url)

        # Build tools payload if provided (OpenAI-compatible format)
        openai_tools = None
        if options.tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in options.tools
            ]

        kwargs: Dict[str, Any] = dict(
            model=model.id,   # endpoint_id 或 doubao-seed-2-0-lite-260215 等模型名
            messages=messages,
        )
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        # Accumulator for streaming tool call fragments
        tool_call_accum: Dict[int, Dict[str, Any]] = {}

        # AsyncArk.chat.completions.stream 返回 OpenAI 兼容的 chunk 格式
        stream_resp = await client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text content fragment
            if delta.content:
                yield {"content": delta.content}

            # Tool call delta fragments
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_call_accum[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_accum[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

        # After stream ends, emit assembled tool_calls if any
        if tool_call_accum:
            assembled = []
            for idx in sorted(tool_call_accum.keys()):
                tc = tool_call_accum[idx]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": tc["arguments"]}
                assembled.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": args,
                })
            yield {"tool_calls": assembled}
