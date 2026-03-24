from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .usage import normalize_usage_payload


def content_chunk(content: str) -> Dict[str, Any]:
    return {"content": content}


def reasoning_chunk(reasoning: str) -> Dict[str, Any]:
    return {"reasoning": reasoning}


def raw_content_chunk(raw_content: str, *, content: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"raw_content": raw_content}
    if content is not None:
        payload["content"] = content
    return payload


def tool_calls_chunk(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"tool_calls": tool_calls}


def provider_state_chunk(namespace: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"provider_state": {namespace: payload}}


def usage_chunk(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {"usage": usage}


def parse_json_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if not arguments:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
    return {"raw": arguments}


__all__ = [
    "content_chunk",
    "normalize_usage_payload",
    "parse_json_arguments",
    "provider_state_chunk",
    "raw_content_chunk",
    "reasoning_chunk",
    "tool_calls_chunk",
    "usage_chunk",
]
