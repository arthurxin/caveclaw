"""
MiniMax Provider adapter.
基于 src/minimax_calling.py 的实现，使用 OpenAI 兼容 API。

Key differences from standard OpenAI:
- 使用 OpenAI 兼容接口，但需要自定义 base_url
- 支持 extra_body={"reasoning_split": True}, 使模型将思考过程包裹在 <think>...</think> 中
- 我们把 <think> 块提取出来，作为独立的 "reasoning" 事件 yield
- Tool calls 遵循标准 OpenAI function_call 结构

.env 配置:
    MINIMAX_API_KEY=sk-xxxxx
    MINIMAX_BASE_URL=https://oneapi.hkgai.net/v1
"""
import os
import re
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions


class MiniMaxProvider:
    """
    Implements the ApiProvider protocol for MiniMax models.
    Communicates via the OpenAI-compatible REST API with custom base_url and
    handles the <think>...</think> reasoning format.

    base_url resolution priority:
      1. model.baseUrl (from models.json, if set)
      2. MINIMAX_BASE_URL env var
      3. default fallback: http://localhost:8080/v1
    """
    api = "minimax-local"

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required: uv add openai")

        key = api_key or os.environ.get("MINIMAX_API_KEY", "EMPTY")
        base_url = (
            model.baseUrl
            or os.environ.get("MINIMAX_BASE_URL")
            or "http://localhost:8080/v1"
        )

        client = AsyncOpenAI(api_key=key, base_url=base_url)

        # Build tools payload (OpenAI format, same as OpenAiProvider)
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
            model=model.id,
            messages=messages,
            extra_body={"reasoning_split": True},  # Enables <think>...</think> wrapping
            temperature=0.6,
            top_p=0.7,
            max_tokens=model.maxTokens,
        )
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        # Accumulators
        tool_call_accum: Dict[int, Dict[str, Any]] = {}
        content_buffer = ""

        # Use standard create(stream=True) to get raw SSE chunks
        stream_resp = await client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Accumulate content for think-block stripping
            if delta.content:
                content_buffer += delta.content

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

        # After stream, parse think block from accumulated content
        if content_buffer:
            think_match = re.search(r"<think>(.*?)</think>", content_buffer, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                if reasoning:
                    yield {"reasoning": reasoning}

            # Strip <think>...</think> and yield clean content
            clean_content = re.sub(r"<think>.*?</think>", "", content_buffer, flags=re.DOTALL).strip()
            if clean_content:
                yield {"content": clean_content}

        # Emit assembled tool_calls if any
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
