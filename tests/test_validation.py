import unittest

from agent_core.llm_provider.validation import ToolValidationError, validate_tool_arguments


class _Tool:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters


class ValidationTests(unittest.TestCase):
    def test_validate_tool_arguments_coerces_basic_scalar_types(self):
        tool = _Tool(
            "coerce_values",
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "ratio": {"type": "number"},
                },
                "required": ["count", "enabled", "ratio"],
            },
        )

        result = validate_tool_arguments(
            tool,
            {"count": "7", "enabled": "true", "ratio": "2.5"},
        )

        self.assertEqual(result, {"count": 7, "enabled": True, "ratio": 2.5})

    def test_validate_tool_arguments_supports_one_of_branch_coercion(self):
        tool = _Tool(
            "branching",
            {
                "type": "object",
                "properties": {
                    "value": {
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "string", "enum": ["auto"]},
                        ]
                    }
                },
                "required": ["value"],
            },
        )

        self.assertEqual(validate_tool_arguments(tool, {"value": "9"})["value"], 9)
        self.assertEqual(validate_tool_arguments(tool, {"value": "auto"})["value"], "auto")

    def test_validate_tool_arguments_enforces_format_when_available(self):
        tool = _Tool(
            "email_tool",
            {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
                "required": ["email"],
            },
        )

        with self.assertRaises(ToolValidationError) as ctx:
            validate_tool_arguments(tool, {"email": "not-an-email"})

        self.assertIn("Validation failed for tool", str(ctx.exception))
        self.assertIn("email", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
