"""
Usage Example: Universal LLM Wrapper
Shows how to wire the ModelRegistry, ModelResolver, and an ApiProvider
together to stream a response through the CaveClaw engine.
"""
import asyncio
import os
from agent_core.llm import ModelRegistry, ModelResolver, api_provider_registry, StreamOptions
from agent_core.llm.providers import OpenAiProvider, AnthropicProvider
from agent_core.types import AgentTool, ToolResult, AgentContext

# 1. Register providers at startup
api_provider_registry.register(OpenAiProvider())
api_provider_registry.register(AnthropicProvider())

# 2. Load models from models.json (or use the builtin fallback)
#    If models.json doesn't exist, the registry still works — providers
#    are just resolved by env var convention (OPENAI_API_KEY, etc.)
registry = ModelRegistry(models_json_path="models.json")
resolver = ModelResolver(registry)

async def run_example():
    # 3. Resolve a model by name + optional thinking level
    #    Supports: "gpt-4o", "anthropic/claude-opus-4-6:high", "openai/gpt-4o"
    model, thinking_level = resolver.find_initial_model(
        cli_model_str="gpt-4o",  # or: "anthropic/claude-3-5-sonnet"
    )
    assert model is not None, "No model found! Check your models.json or env vars."
    print(f"Using model: {model.provider}/{model.id}  (thinking: {thinking_level})")

    # 4. Pick the right provider from the registry via the model's api field
    provider = api_provider_registry.get(model.api)
    assert provider is not None, f"No ApiProvider registered for api type: {model.api}"

    # 5. Build options and messages
    options = StreamOptions()
    options.system_prompt = "You are a helpful assistant."
    options.thinking_level = thinking_level

    messages = [{"role": "user", "content": "Hello! Summarize what CaveClaw is in one sentence."}]

    # 6. Get the api_key from the registry (reads from env by convention)
    api_key = registry.get_api_key(model.provider)

    # 7. Stream the response
    print("\n--- Response ---")
    full_response = ""
    async for chunk in provider.stream(model, messages, options, api_key=api_key):
        if "content" in chunk:
            print(chunk["content"], end="", flush=True)
            full_response += chunk["content"]
        if "tool_calls" in chunk:
            print(f"\n[Tool calls: {chunk['tool_calls']}]")
    print("\n--- End ---")


if __name__ == "__main__":
    asyncio.run(run_example())
