"""
Google Gemini Provider adapter.
Normalizes the Google Generative AI (Gemini) streaming API into the unified
dict format expected by agent_loop.

Gemini's streaming differs from OpenAI/Anthropic:
- Uses google-generativeai SDK (or google-genai)
- Tool calls arrive as function_call parts inside a Content
- Text arrives as text parts
- We handle both via candidates[0].content.parts iteration
"""
import os
import json
from typing import AsyncGenerator, Dict, Any, List, Optional

from ..provider_types import Model
from ..api_registry import StreamOptions


class GoogleProvider:
    """
    Implements the ApiProvider protocol for the Google Gemini API.
    Uses the `google-generativeai` SDK (pip install google-generativeai).

    Supports:
    - google (official Gemini API via generativelanguage.googleapis.com)
    - google-vertex (Vertex AI, requires different auth)
    """
    api = "google-gemini"

    async def stream(
        self,
        model: Model,
        messages: List[Dict[str, Any]],
        options: StreamOptions,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package is required: uv add google-generativeai")

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No Google API key found. Set GEMINI_API_KEY in your .env")

        genai.configure(api_key=key)

        # Build Gemini tool declarations if provided
        gemini_tools = None
        if options.tools:
            # Gemini accepts function declarations via Tool objects
            function_declarations = []
            for t in options.tools:
                # Filter out unsupported JSON Schema keywords Gemini doesn't accept
                parameters = _sanitize_schema(t.parameters)
                function_declarations.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                })
            gemini_tools = [genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(**fd) for fd in function_declarations
                ]
            )]

        # Convert messages from OpenAI format to Gemini Content format
        gemini_history, last_user_message = _convert_messages(messages)

        # Set up the model client
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=model.maxTokens,
        )

        gemini_model = genai.GenerativeModel(
            model_name=model.id,
            system_instruction=options.system_prompt or None,
            generation_config=generation_config,
            tools=gemini_tools,
        )

        chat = gemini_model.start_chat(history=gemini_history)

        # Stream the response
        response = await chat.send_message_async(last_user_message, stream=True)

        async for chunk in response:
            for part in chunk.parts:
                # Text content fragment
                if hasattr(part, "text") and part.text:
                    yield {"content": part.text}

                # Function call (tool call)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Convert Gemini's MapComposite args to a regular dict
                    try:
                        args = dict(fc.args) if fc.args else {}
                    except Exception:
                        args = {}
                    yield {
                        "tool_calls": [{
                            "id": f"call_{fc.name}_{id(fc)}",  # Gemini doesn't give call IDs
                            "name": fc.name,
                            "arguments": args,
                        }]
                    }


def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove JSON Schema keywords that Gemini's function declaration doesn't support.
    Gemini only accepts a subset of JSON Schema.
    """
    UNSUPPORTED_KEYS = {"$schema", "additionalProperties", "default", "examples", "const"}
    if not isinstance(schema, dict):
        return schema
    return {
        k: _sanitize_schema(v) if isinstance(v, dict) else v
        for k, v in schema.items()
        if k not in UNSUPPORTED_KEYS
    }


def _convert_messages(
    messages: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], str]:
    """
    Convert an OpenAI-format message history into Gemini's Content format.
    Returns (history, last_user_message_text).
    Gemini requires alternating user/model turns, and system messages are handled via system_instruction.
    """
    history = []
    last_user_msg = ""

    # Skip system messages (handled via system_instruction)
    conversation = [m for m in messages if m.get("role") != "system"]

    # The last user message is sent separately via send_message
    # Everything before is the history
    if conversation and conversation[-1]["role"] == "user":
        last_user_msg = conversation[-1].get("content", "")
        conversation = conversation[:-1]

    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Map OpenAI roles to Gemini roles
        gemini_role = "user" if role in ("user", "tool") else "model"
        history.append({
            "role": gemini_role,
            "parts": [{"text": content}],
        })

    return history, last_user_msg
