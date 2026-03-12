"""
OpenAI Provider adapter.
Normalizes the OpenAI Streaming API response chunks (tool_calls fragments)
into the unified dict format expected by agent_loop.
"""
import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions

class OpenAiProvider:
    """
    Implements the ApiProvider protocol for the OpenAI Completions API.
    Handles both regular text content and streamed tool_calls delta assembly.

    Supports:
    - openai (official api)
    - azure-openai-responses
    - openai-compatible APIs (DeepSeek, Groq, etc.) via custom baseUrl
    """
    api = "openai-chat"

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

        key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = model.baseUrl or os.environ.get("OPENAI_BASE_URL")

        client = AsyncOpenAI(api_key=key, base_url=base_url or None)

        # Build tools payload if provided
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
            stream=True,
        )
        if openai_tools:
            kwargs["tools"] = openai_tools
        if options.thinking_level in ("high", "xhigh") and model.reasoning:
            kwargs["reasoning_effort"] = "high"

        # Accumulator for streaming tool call fragments
        # OpenAI streams tool_calls as per-index deltas; we reassemble them here.
        tool_call_accum: Dict[int, Dict[str, Any]] = {}

        async with client.chat.completions.stream(**kwargs) as stream_resp:
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
