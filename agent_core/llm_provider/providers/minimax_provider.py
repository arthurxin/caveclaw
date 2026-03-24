"""
MiniMax Provider adapter.
基于 legacy MiniMax 封装的思路，使用 OpenAI 兼容 API。

Key differences from standard OpenAI:
- 使用 OpenAI 兼容接口，但需要自定义 base_url
- 支持 extra_body={"reasoning_split": True}, 使模型将思考过程包裹在 <think>...</think> 中
- 我们把 <think> 块提取出来，作为独立的 "reasoning" 事件 yield
- Tool calls 遵循标准 OpenAI function_call 结构

.env 配置:
    MINIMAX_API_KEY=sk-xxxxx
    MINIMAX_BASE_URL=https://oneapi.hkgai.net/v1
"""
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions, merge_request_headers, maybe_override_payload, resolve_effective_max_tokens
from ..compat import (
    build_openai_compatible_tools,
    build_reasoning_payload,
    normalize_openai_compatible_messages,
    resolve_max_tokens_field,
    supports_usage_in_streaming,
)
from ..message_codec import MiniMaxMessageCodec
from ..normalized_chunks import (
    normalize_usage_payload,
    parse_json_arguments,
    raw_content_chunk,
    reasoning_chunk,
    tool_calls_chunk,
    usage_chunk,
)


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
    message_codec = MiniMaxMessageCodec()

    @staticmethod
    def _extract_reasoning_and_content(content_buffer: str) -> Dict[str, Optional[str]]:
        reasoning = None
        think_match = re.search(r"<think>(.*?)</think>", content_buffer, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip() or None

        clean_content = re.sub(r"<think>.*?</think>", "", content_buffer, flags=re.DOTALL).strip()
        return {
            "raw_content": content_buffer,
            "reasoning": reasoning,
            "content": clean_content,
        }

    @staticmethod
    def _assemble_tool_calls(tool_call_accum: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        assembled: List[Dict[str, Any]] = []
        for idx in sorted(tool_call_accum.keys()):
            tool_call = tool_call_accum[idx]
            arguments = parse_json_arguments(tool_call["arguments"])
            assembled.append(
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "arguments": arguments,
                }
            )
        return assembled

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

        key = api_key or "EMPTY"
        base_url = (
            model.baseUrl
            or "http://localhost:8080/v1"
        )
        default_headers = merge_request_headers(model, options) or None

        client = AsyncOpenAI(api_key=key, base_url=base_url, default_headers=default_headers)

        provider_messages = normalize_openai_compatible_messages(model, messages)

        openai_tools = build_openai_compatible_tools(model, options.tools)

        kwargs: Dict[str, Any] = dict(
            model=model.id,
            messages=provider_messages,
            extra_body={"reasoning_split": True},  # Enables <think>...</think> wrapping
            temperature=options.temperature if options.temperature is not None else 0.6,
            top_p=0.7,
        )
        kwargs[resolve_max_tokens_field(model)] = resolve_effective_max_tokens(model, options)
        if supports_usage_in_streaming(model):
            kwargs["stream_options"] = {"include_usage": True}
        kwargs.update(build_reasoning_payload(model, options.thinking_level))
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        if options.metadata:
            kwargs["metadata"] = dict(options.metadata)
        if options.session_id:
            metadata = dict(kwargs.get("metadata") or {})
            metadata.setdefault("session_id", options.session_id)
            kwargs["metadata"] = metadata
        kwargs = dict(await maybe_override_payload(kwargs, model, options))

        # Accumulators
        tool_call_accum: Dict[int, Dict[str, Any]] = {}
        content_buffer = ""

        # Use standard create(stream=True) to get raw SSE chunks
        stream_resp = await client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream_resp:
            if getattr(chunk, "usage", None):
                usage = normalize_usage_payload(
                    input_tokens=getattr(chunk.usage, "prompt_tokens", None),
                    output_tokens=getattr(chunk.usage, "completion_tokens", None),
                    total_tokens=getattr(chunk.usage, "total_tokens", None),
                )
                if usage:
                    yield usage_chunk(usage)

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
            parsed_content = self._extract_reasoning_and_content(content_buffer)
            if parsed_content["reasoning"]:
                yield reasoning_chunk(parsed_content["reasoning"])
            yield raw_content_chunk(
                parsed_content["raw_content"],
                content=parsed_content["content"],
            )
        # Emit assembled tool_calls if any
        if tool_call_accum:
            yield tool_calls_chunk(self._assemble_tool_calls(tool_call_accum))
