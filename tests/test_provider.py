import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core.llm.registry import ModelRegistry
from agent_core.llm.resolver import ModelResolver
from agent_core.llm.api_registry import api_provider_registry, StreamOptions
from agent_core.llm.providers import OpenAiProvider, AnthropicProvider, GoogleProvider, MiniMaxProvider, ArkProvider

# Register providers
api_provider_registry.register(OpenAiProvider())
api_provider_registry.register(AnthropicProvider())
api_provider_registry.register(GoogleProvider())
api_provider_registry.register(MiniMaxProvider())
api_provider_registry.register(ArkProvider())

async def test_streaming(model_uri: str):
    print(f"\n--- Testing Model: {model_uri} ---")
    
    # 1. Initialize Registry and Resolver
    registry = ModelRegistry()
    resolver = ModelResolver(registry)
    
    # 2. Resolve Model
    model, thinking_level = resolver.resolve(model_uri)
    if not model:
        print(f"Error: Could not resolve model '{model_uri}'")
        return

    print(f"Resolved Model: {model.provider}/{model.id}")
    print(f"Thinking Level: {thinking_level}")
    print(f"API Type: {model.api}")
    
    # 3. Get API Key
    api_key = registry.get_api_key(model.provider)
    print(f"Debug - Provider: {model.provider}")
    print(f"Debug - API Key Found: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"Debug - API Key Starts With: {api_key[:8]}...")
    
    # Get the provider to check its base_url logic (simulated by looking at env)
    if model.provider == "minimax":
        print(f"Debug - MINIMAX_BASE_URL (env): {os.environ.get('MINIMAX_BASE_URL')}")
        print(f"Debug - Model.baseUrl: {model.baseUrl}")

    if not api_key:
        print(f"Error: No API key found for provider '{model.provider}'")
        print(f"Please set {model.provider.upper().replace('-', '_')}_API_KEY in .env")
        return

    # 4. Get Provider and Stream
    provider = api_provider_registry.get(model.api)
    if not provider:
        print(f"Error: No provider registered for API '{model.api}'")
        return

    messages = [
        {"role": "user", "content": "你好，请自我介绍一下，并告诉我你当前的思考深度。"}
    ]
    
    options = StreamOptions()
    options.thinking_level = thinking_level
    
    print("Streaming Response:\n" + "-"*20)
    
    try:
        async for chunk in provider.stream(
            model=model,
            messages=messages,
            options=options,
            api_key=api_key
        ):
            if "reasoning" in chunk:
                # Print reasoning in dim color if possible, or just bracketed
                print(f"[Reasoning]: {chunk['reasoning']}", end="", flush=True)
            if "content" in chunk:
                print(chunk["content"], end="", flush=True)
            if "tool_calls" in chunk:
                print(f"\n[Tool Call]: {chunk['tool_calls']}")
                
        print("\n" + "-"*20 + "\nSuccess!")
    except Exception as e:
        print(f"\nError during streaming: {str(e)}")

if __name__ == "__main__":
    load_dotenv()
    
    # Default to minimax for the user
    target_model = sys.argv[1] if len(sys.argv) > 1 else "minimax/abab6.5s-chat"
    
    asyncio.run(test_streaming(target_model))
