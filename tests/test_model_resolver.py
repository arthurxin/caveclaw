import os
import unittest

from agent_core.llm import ModelRegistry, ModelResolver


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelResolverTests(unittest.TestCase):
    def setUp(self):
        models_path = os.path.join(PROJECT_ROOT, "models.json")
        self.registry = ModelRegistry(models_json_path=models_path)
        self.resolver = ModelResolver(self.registry)

    def test_resolve_explicit_thinking_level(self):
        model, thinking = self.resolver.resolve("openai/gpt-4o:high")

        self.assertIsNotNone(model)
        self.assertEqual(model.provider, "openai")
        self.assertEqual(model.id, "gpt-4o")
        self.assertEqual(thinking, "high")

    def test_resolve_fallback_model_for_known_provider(self):
        model, thinking = self.resolver.resolve("openai/custom-model:low")

        self.assertIsNotNone(model)
        self.assertEqual(model.provider, "openai")
        self.assertEqual(model.id, "custom-model")
        self.assertEqual(thinking, "low")

    def test_find_initial_model_uses_current_openai_default(self):
        model, thinking = self.resolver.find_initial_model(cli_model_str=None)

        self.assertIsNotNone(model)
        self.assertEqual(model.provider, "openai")
        self.assertEqual(model.id, "gpt-5.4")
        self.assertEqual(thinking, "off")


if __name__ == "__main__":
    unittest.main()
