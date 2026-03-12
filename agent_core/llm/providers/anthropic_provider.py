"""
Anthropic Provider adapter.
Normalizes Anthropic's streaming API (content_block_delta events, tool_use blocks)
into the unified dict format expected by agent_loop.

Anthropic's streaming is different from OpenAI:
- Text arrives via content_block_delta events with type="text_delta"
- Tool calls arrive via content_block_delta with type="input_json_delta" (JSON fragments)
- We must track which content block we are in via content_block_start events
"""
import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions


class AnthropicProvider:
    """
    Implements the ApiProvider protocol for the Anthropic Messages API.
    Handles streaming of text and tool_use content block events.

    Supports:
    - anthropic (official api, claude-* models)
    - amazon-bedrock (via boto3 with Anthropic models, handled separately)
    """
    api = "anthropic-messages"

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required: uv add anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        base_url = model.baseUrl or os.environ.get("ANTHROPIC_BASE_URL")

        client_kwargs: Dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = anthropic.AsyncAnthropic(**client_kwargs)

        # Build tools payload if provided
        anthropic_tools = None
        if options.tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in options.tools
            ]

        # Separate system prompt from conversation history
        system_prompt = options.system_prompt
        conversation = [m for m in messages if m.get("role") != "system"]

        kwargs: Dict[str, Any] = dict(
            model=model.id,
            max_tokens=model.maxTokens,
            messages=conversation,
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # Thinking (extended thinking) support
        if options.thinking_level in ("high", "xhigh") and model.reasoning:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}

        # ---------------------------------------------------------------
        # Track state across streaming events:
        # content_block_start tells us what type of block started (text / tool_use)
        # content_block_delta carries the actual payload
        # ---------------------------------------------------------------
        current_block_type: Optional[str] = None  # "text" | "tool_use"
        current_tool: Optional[Dict[str, Any]] = None  # accumulates tool id/name/json
        current_tool_json: str = ""

        async with client.messages.stream(**kwargs) as stream_resp:
            async for event in stream_resp:
                event_type = event.type

                if event_type == "content_block_start":
                    block = event.content_block
                    current_block_type = block.type
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and delta.text:
                        yield {"content": delta.text}
                    elif delta.type == "input_json_delta" and delta.partial_json:
                        current_tool_json += delta.partial_json

                elif event_type == "content_block_stop":
                    if current_block_type == "tool_use" and current_tool:
                        # Emit the fully assembled tool call
                        try:
                            args = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            args = {"raw": current_tool_json}
                        yield {
                            "tool_calls": [{
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "arguments": args,
                            }]
                        }
                        current_tool = None
                        current_tool_json = ""
                    current_block_type = None
