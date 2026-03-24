from __future__ import annotations

from typing import Any, Dict, Optional

from .provider_types import Model


def _read_usage_int(payload: Dict[str, Any], canonical_key: str, legacy_key: str) -> int:
    value = payload.get(legacy_key)
    if value is None:
        value = payload.get(canonical_key)
    if value is None:
        return 0
    return int(value)


def calculate_usage_cost(model: Model, usage: Dict[str, Any]) -> Dict[str, float]:
    input_tokens = _read_usage_int(usage, "input", "input_tokens")
    output_tokens = _read_usage_int(usage, "output", "output_tokens")
    cache_read_tokens = _read_usage_int(usage, "cacheRead", "cache_read_tokens")
    cache_write_tokens = _read_usage_int(usage, "cacheWrite", "cache_write_tokens")

    input_cost = (model.cost.input / 1_000_000) * input_tokens
    output_cost = (model.cost.output / 1_000_000) * output_tokens
    cache_read_cost = (model.cost.cacheRead / 1_000_000) * cache_read_tokens
    cache_write_cost = (model.cost.cacheWrite / 1_000_000) * cache_write_tokens
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return {
        "input": input_cost,
        "output": output_cost,
        "cacheRead": cache_read_cost,
        "cacheWrite": cache_write_cost,
        "total": total_cost,
    }


def finalize_usage_payload(usage: Dict[str, Any], model: Optional[Model] = None) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        key: value
        for key, value in usage.items()
        if key not in {
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
            "totalTokens",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "cost",
        }
    }

    input_tokens = _read_usage_int(usage, "input", "input_tokens")
    output_tokens = _read_usage_int(usage, "output", "output_tokens")
    cache_read_tokens = _read_usage_int(usage, "cacheRead", "cache_read_tokens")
    cache_write_tokens = _read_usage_int(usage, "cacheWrite", "cache_write_tokens")

    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = usage.get("totalTokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens + cache_read_tokens + cache_write_tokens

    normalized.update(
        {
            "input": input_tokens,
            "output": output_tokens,
            "cacheRead": cache_read_tokens,
            "cacheWrite": cache_write_tokens,
            "totalTokens": int(total_tokens),
            # Keep legacy keys for compatibility while the rest of the codebase converges.
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "total_tokens": int(total_tokens),
        }
    )

    normalized["cost"] = calculate_usage_cost(model, normalized) if model is not None else {
        "input": 0.0,
        "output": 0.0,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
        "total": 0.0,
    }
    return normalized


def normalize_usage_payload(
    *,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    if input_tokens is not None:
        payload["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        payload["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        payload["total_tokens"] = int(total_tokens)
    if cache_read_tokens is not None:
        payload["cache_read_tokens"] = int(cache_read_tokens)
    if cache_write_tokens is not None:
        payload["cache_write_tokens"] = int(cache_write_tokens)
    if extra:
        payload.update(extra)
    if not payload:
        return None
    return finalize_usage_payload(payload)


__all__ = [
    "calculate_usage_cost",
    "finalize_usage_payload",
    "normalize_usage_payload",
]
