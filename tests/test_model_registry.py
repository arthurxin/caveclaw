import json
import os
import tempfile
import unittest

from agent_core.llm_provider import ModelRegistry


class ModelRegistryValidationTests(unittest.TestCase):
    def _write_models_json(self, payload):
        temp_dir = tempfile.TemporaryDirectory()
        models_path = os.path.join(temp_dir.name, "models.json")
        with open(models_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.addCleanup(temp_dir.cleanup)
        return models_path

    def test_registry_records_validation_errors_but_keeps_valid_models(self):
        models_path = self._write_models_json(
            {
                "providers": {
                    "valid": {
                        "api": "openai-chat",
                        "models": [{"id": "good-model"}],
                    },
                    "broken": {
                        "api": "azure-responses",
                        "models": [{"name": "missing id"}],
                    },
                }
            }
        )

        registry = ModelRegistry(models_json_path=models_path)

        self.assertEqual(len(registry.get_all()), 1)
        self.assertEqual(registry.get_all()[0].provider, "valid")
        self.assertTrue(registry.validation_errors)
        self.assertIn("models[0].id", registry.load_error)

    def test_registry_rejects_non_mapping_headers(self):
        models_path = self._write_models_json(
            {
                "providers": {
                    "broken": {
                        "api": "openai-chat",
                        "headers": ["bad"],
                        "models": [{"id": "gpt"}],
                    }
                }
            }
        )

        registry = ModelRegistry(models_json_path=models_path)

        self.assertEqual(registry.get_all(), [])
        self.assertIn("headers", registry.load_error)


if __name__ == "__main__":
    unittest.main()
