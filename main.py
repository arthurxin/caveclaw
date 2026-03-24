from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

from dotenv import load_dotenv

from Agent_Prototype.mainchat_agent import MainChatAgent, MainChatAgentConfig
from agent_core import AgentContext, Message
from agent_core.core.inspector import PythonRuntimeInspector
from agent_core.llm_provider import Model, ModelRegistry, ModelResolver, get_env_api_key, register_builtin_providers


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Agent_Prototype" / "data"
CSV_PATH = DATA_DIR / "fake_sales.csv"
DOTENV_PATH = PROJECT_ROOT / ".env"


def ensure_fake_csv(csv_path: Path = CSV_PATH) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["region", "sales", "orders"],
        ["North", "120", "4"],
        ["South", "85", "3"],
        ["East", "140", "5"],
        ["West", "95", "4"],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return csv_path


def print_csv_preview(csv_path: Path) -> None:
    print(f"CSV path: {csv_path}")
    print("CSV preview:")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            print(line.rstrip())


class PrototypeMainChatAgent(MainChatAgent):
    def __init__(self, *, model: Model):
        super().__init__(model=model)

    def build_system_prompt(self) -> str:
        return (
            "You are the prototype main chat agent. "
            "When the user asks about a CSV or dataframe task, prefer replying with a fenced "
            "python block that performs the analysis before giving a final answer. "
            "Important: fenced python blocks are executed inside the user's local runtime. "
            "The later python execution result, synced runtime variables, and runtime snapshots "
            "are authoritative local facts, not hypothetical examples. "
            "If execution output already reveals the schema or the answer, use it directly. "
            "If execution fails, read stdout/stderr/runtime snapshot carefully, correct the code, "
            "and continue only as needed. "
            "Do not keep saying that you cannot access the local file after local execution output "
            "or runtime snapshots have been provided. "
            "Once you have enough information, answer concisely and stop."
        )

    def configure_config(self, config: MainChatAgentConfig) -> None:
        config.python_program_execution = True
        config.python_program_backend = "python"
        config.inspector = PythonRuntimeInspector()

def initialize_real_model(
    *,
    models_json_path: Path = PROJECT_ROOT / "models.json",
    model_uri: Optional[str] = None,
) -> Model:
    load_dotenv(DOTENV_PATH)
    register_builtin_providers()
    registry = ModelRegistry(str(models_json_path))
    resolver = ModelResolver(registry)
    resolved_model, thinking_level = resolver.find_initial_model(model_uri)
    if resolved_model is None:
        raise RuntimeError(
            "Could not resolve a model from models.json or environment defaults. "
            "Check models.json, CAVECLAW_DEFAULT_PROVIDER, and CAVECLAW_DEFAULT_MODEL."
        )

    api_key = get_env_api_key(resolved_model.provider)
    if not api_key:
        expected_env = f"{resolved_model.provider.upper().replace('-', '_')}_API_KEY"
        raise RuntimeError(
            f"No API key found for provider '{resolved_model.provider}'. "
            f"Please set {expected_env} in {DOTENV_PATH} or the current environment."
        )

    print(
        "Resolved model:",
        f"{resolved_model.provider}/{resolved_model.id}",
        f"(thinking={thinking_level})",
    )
    return resolved_model


async def print_runtime_snapshot(agent: PrototypeMainChatAgent) -> None:
    inspector = PythonRuntimeInspector()
    context = AgentContext(
        messages=list(agent.agent.state.messages),
        runtime=agent.agent.state.runtime,
    )
    print("\nRuntime raw variables:")
    for key, variable in agent.agent.state.runtime.variables.items():
        print(f"- {key}: kind={variable.kind} version={variable.version}")
        print(f"  raw_value={variable.raw_value!r}")

    snapshot = await inspector.capture_state(context)
    print("\nRuntime snapshot:")
    for entry in snapshot.entries:
        print(f"- {entry.key} (v{entry.version})")
        for block in entry.summary_blocks:
            if hasattr(block, "text"):
                print(block.text)


async def run_demo(*, model_uri: Optional[str] = None) -> None:
    csv_path = ensure_fake_csv()
    print_csv_preview(csv_path)

    model = initialize_real_model(model_uri=model_uri)
    agent = PrototypeMainChatAgent(model=model)
    print("\nAgent conversation:")
    try:
        async for event in agent.handle_user_input(
            f"Please analyze the CSV at {csv_path} and tell me the top sales region.",
        ):
            if event.type == "message_end" and getattr(event.message, "role", None) == "assistant":
                print(f"[assistant] {event.message.content}")
            elif event.type == "python_program_execution_success":
                print(f"[python lane] synced variables: {event.get('synced_variables')}")
                print(f"[python lane] namespace before keys: {event.get('namespace_before_keys')}")
                print(f"[python lane] namespace after keys: {event.get('namespace_after_keys')}")
                skipped_variables = event.get("skipped_variables") or {}
                if skipped_variables:
                    print(f"[python lane] skipped variables: {skipped_variables}")
                python_result = event.result
                if python_result is not None and getattr(python_result, "stdout", None):
                    print(f"[python stdout] {python_result.stdout.strip()}")
                if python_result is not None and getattr(python_result, "error", None):
                    print(f"[python error] {python_result.error}")
    except Exception as error:
        print("\nAgent run failed.")
        print(f"Model: {model.provider}/{model.id}")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")
        if "Connection error" in str(error) or "ConnectError" in repr(error):
            print(
                "Hint: this usually means the selected provider endpoint is unreachable from the "
                "current environment, or DNS/network access is blocked."
            )
        await print_runtime_snapshot(agent)
        return

    await print_runtime_snapshot(agent)


if __name__ == "__main__":
    target_model_uri = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CAVECLAW_MAIN_MODEL")
    asyncio.run(run_demo(model_uri=target_model_uri))
