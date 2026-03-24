from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Optional

try:
    from jsonschema import Draft202012Validator, FormatChecker
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

    _JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    Draft202012Validator = None
    FormatChecker = None
    JsonSchemaValidationError = Exception
    _JSONSCHEMA_AVAILABLE = False


class ToolValidationError(ValueError):
    pass


def normalize_tool_arguments(arguments: Dict[str, Any] | str) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(f"Invalid tool arguments JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ToolValidationError("Tool arguments must decode to a JSON object.")
        return payload
    raise ToolValidationError("Tool arguments must be a mapping or JSON object string.")


def validate_tool_arguments(schema_holder: Any, arguments: Dict[str, Any] | str) -> Dict[str, Any]:
    normalized = normalize_tool_arguments(arguments)
    schema = getattr(schema_holder, "parameters", None)
    if not isinstance(schema, dict) or not schema:
        return normalized

    effective_schema = _coerce_schema(schema)
    coerced = _coerce_value_for_schema(effective_schema, copy.deepcopy(normalized))

    if _JSONSCHEMA_AVAILABLE:
        validator = Draft202012Validator(effective_schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(coerced), key=_error_sort_key)
        if not errors:
            return coerced
        raise ToolValidationError(_format_jsonschema_errors(schema_holder, normalized, errors))

    _validate_schema(effective_schema, coerced, path="$")
    return coerced


def _coerce_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if "type" in schema or "properties" in schema or "required" in schema or "oneOf" in schema or "anyOf" in schema:
        return schema
    return {
        "type": "object",
        "properties": dict(schema),
    }


def _coerce_value_for_schema(schema: Dict[str, Any], value: Any) -> Any:
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list) and variants:
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                candidate = _coerce_value_for_schema(variant, copy.deepcopy(value))
                if _value_matches_schema(variant, candidate):
                    return candidate

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        candidate = copy.deepcopy(value)
        for part in all_of:
            if isinstance(part, dict):
                candidate = _coerce_value_for_schema(part, candidate)
        return candidate

    expected_type = schema.get("type")
    expected_types = [expected_type] if isinstance(expected_type, str) else [t for t in expected_type or [] if isinstance(t, str)]

    if "object" in expected_types and isinstance(value, dict):
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            coerced = dict(value)
            for key, child_schema in properties.items():
                if key in coerced and isinstance(child_schema, dict):
                    coerced[key] = _coerce_value_for_schema(child_schema, coerced[key])
            return coerced
        return value

    if "array" in expected_types and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [_coerce_value_for_schema(item_schema, item) for item in value]
        return value

    if "boolean" in expected_types:
        coerced = _coerce_boolean(value)
        if coerced is not _NO_VALUE:
            return coerced

    if "integer" in expected_types:
        coerced = _coerce_integer(value)
        if coerced is not _NO_VALUE:
            return coerced

    if "number" in expected_types:
        coerced = _coerce_number(value)
        if coerced is not _NO_VALUE:
            return coerced

    if "null" in expected_types and isinstance(value, str) and value.strip().lower() in {"null", "none"}:
        return None

    if "string" in expected_types and value is not None and not isinstance(value, str):
        if isinstance(value, (bool, int, float)):
            return str(value)

    return value


_NO_VALUE = object()


def _coerce_boolean(value: Any) -> bool | object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return _NO_VALUE


def _coerce_integer(value: Any) -> int | object:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return _NO_VALUE


def _coerce_number(value: Any) -> float | int | object:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            number = float(stripped)
        except ValueError:
            return _NO_VALUE
        return int(number) if number.is_integer() else number
    return _NO_VALUE


def _value_matches_schema(schema: Dict[str, Any], value: Any) -> bool:
    if _JSONSCHEMA_AVAILABLE:
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        return validator.is_valid(value)
    try:
        _validate_schema(schema, value, path="$")
    except ToolValidationError:
        return False
    return True


def _error_sort_key(error: JsonSchemaValidationError) -> tuple[int, str]:
    path = ".".join(str(part) for part in error.absolute_path)
    return (len(path), path)


def _format_jsonschema_errors(
    schema_holder: Any,
    arguments: Dict[str, Any],
    errors: Iterable[JsonSchemaValidationError],
) -> str:
    tool_name = getattr(schema_holder, "name", "tool")
    lines: List[str] = []
    for error in errors:
        path = "$"
        if error.absolute_path:
            path = "$." + ".".join(str(part) for part in error.absolute_path)
        elif error.validator == "required":
            missing = error.validator_value
            if isinstance(missing, list):
                missing_name = next((name for name in missing if name not in error.instance), "unknown")
                path = f"$.{missing_name}"
        lines.append(f"  - {path}: {error.message}")

    joined = "\n".join(dict.fromkeys(lines))
    return (
        f'Validation failed for tool "{tool_name}":\n'
        f"{joined}\n\n"
        f"Received arguments:\n{json.dumps(arguments, indent=2, ensure_ascii=False)}"
    )


def _validate_schema(schema: Dict[str, Any], value: Any, *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be an object.")
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    raise ToolValidationError(f"{path}.{key} is required.")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    _validate_schema(child_schema, value[key], path=f"{path}.{key}")
        return

    if expected_type == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array.")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item_schema, item, path=f"{path}[{index}]")
        return

    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string.")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be an integer.")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a number.")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a boolean.")

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values and value not in enum_values:
        raise ToolValidationError(f"{path} must be one of {enum_values}.")


__all__ = ["ToolValidationError", "normalize_tool_arguments", "validate_tool_arguments"]
