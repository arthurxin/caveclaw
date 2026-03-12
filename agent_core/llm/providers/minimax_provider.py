"""
MiniMax Provider adapter.
Based on MinimaxCalling from src/minimax_calling.py.

Key differences from standard OpenAI:
- Uses OpenAI-compatible API but with a custom base_url
- Supports `extra_body={"reasoning_split": True}` which causes the model to
  wrap reasoning tokens in <think>...</think> tags inside the content field
- We strip out <think> blocks and yield them as a separate "reasoning" event
- Tool calls follow the standard OpenAI function_call structure, so assembly
  logic is the same as OpenAiProvider
"""
import os
import re
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions


class MiniMaxProvider:
    """
    Implements the ApiProvider protocol for local/self-hosted MiniMax models.
    Communicates via the OpenAI-compatible REST API with custom base_url and
    handles the <think>...</think> reasoning format.

    Expected .env configuration:
        MINIMAX_API_KEY=...
        MINIMAX_BASE_URL=http://localhost:8080/v1  (or remote proxy)
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
            stream=True,
            extra_body={"reasoning_split": True},  # Enables <think>...</think> wrapping
            temperature=0.6,
            top_p=0.7,
            max_tokens=model.maxTokens,
        )
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        # Accumulate tool_call delta fragments (same pattern as OpenAiProvider)
        tool_call_accum: Dict[int, Dict[str, Any]] = {}

        # Buffer to accumulate raw content and detect <think> blocks
        content_buffer = ""

        async with client.chat.completions.stream(**kwargs) as stream_resp:
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

        # After stream, parse the think block out of the accumulated content
        if content_buffer:
            think_match = re.search(r"<think>(.*?)</think>", content_buffer, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                # Yield reasoning as a discrete event so the engine can log/display it
                if reasoning:
                    yield {"reasoning": reasoning}

            # Strip <think>...</think> from visible content and yield
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
