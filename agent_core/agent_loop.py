import os
import uuid
from typing import AsyncGenerator, List, Callable, Optional, Dict, Any, Tuple

from dotenv import load_dotenv

from .types import (
    AgentMessage, AgentLoopConfig, AgentTool, AgentEvent,
    Message, AssistantMessage, ToolCall, ToolResultMessage,
    AgentContext
)

# Load environment variables
load_dotenv()

async def _stream_assistant_response(
    messages: List[Message],
    stream_fn: Callable[..., AsyncGenerator[Any, None]],
    config: AgentLoopConfig
) -> AssistantMessage:
    """Wrapper that reads from the provided stream_fn and constructs standard AssistantMessage."""
    assistant_msg = AssistantMessage(role="assistant", content="")
    
    # In a full real-world app, stream_fn yields diffs/events 
    # matching the underlying AI provider.
    async for chunk in stream_fn(messages):
        if "content" in chunk:
            assistant_msg.content += chunk["content"]
        elif "tool_calls" in chunk:
            if not assistant_msg.tool_calls:
                assistant_msg.tool_calls = []
            for tc in chunk["tool_calls"]:
                assistant_msg.tool_calls.append(ToolCall(**tc))
                
    return assistant_msg


async def run_loop(
    context_messages: List[AgentMessage],
    tools: List[AgentTool],
    config: AgentLoopConfig,
    stream_fn: Callable[..., AsyncGenerator[Any, None]]
) -> AsyncGenerator[AgentEvent, None]:
    """The core state machine loop (enhanced with Phase 1 safety limits and structural events)."""
    
    # Read limits from Config first, then .env, then fallback to defaults
    env_max_rounds = int(os.environ.get("MAX_ROUNDS", "20"))
    env_max_consec_failures = int(os.environ.get("MAX_CONSECUTIVE_FAILURES", "5"))
    
    max_rounds = getattr(config, "max_rounds", env_max_rounds) or env_max_rounds
    max_consecutive_failures = getattr(config, "max_consecutive_tool_failures", env_max_consec_failures) or env_max_consec_failures
    
    # Initialize the runtime context for the tools
    agent_context = AgentContext(messages=context_messages, shared_memory={})
    
    loop_event_id = str(uuid.uuid4())
    yield AgentEvent(type="agent_start", event_id=loop_event_id)
    
    round_count = 0
    consecutive_failures = 0
    
    while True:
        round_count += 1
        turn_event_id = str(uuid.uuid4())
        
        # Consolidation check - if we hit max rounds
        if round_count > max_rounds:
            yield AgentEvent(
                type="consolidation_required", 
                parent_id=loop_event_id, 
                data={"reason": "max_rounds_reached", "max_rounds": max_rounds}
            )
            # Future phases will implement actual consolidation here
            break
            
        # Fail-Safe Quotas check
        if consecutive_failures >= max_consecutive_failures:
            yield AgentEvent(
                type="human_intervention_required", 
                parent_id=loop_event_id, 
                data={"reason": "max_consecutive_failures", "consecutive_failures": consecutive_failures}
            )
            break
            
        # 1. Transform Context (e.g. context window pruning)
        if hasattr(config, "transform_context"):
            context_messages = await config.transform_context(context_messages)
            
        # 2. Convert to LLM format (filter out CustomMessages/UI elements)
        llm_messages = await config.convert_to_llm(context_messages)
        
        yield AgentEvent(
            type="turn_start", 
            event_id=turn_event_id, 
            parent_id=loop_event_id, 
            data={"round": round_count}
        )
        
        # 3. Stream Response
        assistant_msg = await _stream_assistant_response(llm_messages, stream_fn, config)
        context_messages.append(assistant_msg)
        
        # 4. Check array of Tool Calls
        if not assistant_msg.tool_calls:
            # End of turn, no tools -> Fully finished thinking
            yield AgentEvent(type="turn_end", parent_id=turn_event_id, data={"message": assistant_msg})
            
            # Reset consecutive errors since it successfully concluded a thought
            consecutive_failures = 0
            
            # 5. Check Follow Up messages queue
            if hasattr(config, "get_followup_messages"):
                followup = await config.get_followup_messages()
                if followup:
                    context_messages.extend(followup)
                    continue
            break # Fully finished
            
        # 6. Execute Tools (Observation-only Steering built-in to loop)
        steering_interrupted = False
        
        for tool_call in assistant_msg.tool_calls:
            tool_event_id = str(uuid.uuid4())
            yield AgentEvent(
                type="tool_execution_start", 
                event_id=tool_event_id, 
                parent_id=turn_event_id, 
                data={"tool_call": tool_call}
            )
            
            tool = next((t for t in tools if t.name == tool_call.name), None)
            
            if not tool:
                error_msg = f"Error: Tool {tool_call.name} not found."
                context_messages.append(ToolResultMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=error_msg
                ))
                consecutive_failures += 1
                yield AgentEvent(type="tool_execution_error", parent_id=tool_event_id, data={"error": error_msg})
            else:
                try:
                    # Execute tool with context and capture delta
                    result = await tool.execute(tool_call.id, tool_call.arguments, agent_context)
                    
                    # Merge state delta safely if returned
                    if getattr(result, "state_delta", None):
                        # Use dict.update for direct key overwrites (shallow merge).
                        # Future improvements can implement deep merge if dictated by design.
                        agent_context.shared_memory.update(result.state_delta)
                        
                    context_messages.append(ToolResultMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=result.content
                    ))
                    # Reset failure count on successful execution
                    consecutive_failures = 0
                    yield AgentEvent(
                        type="tool_execution_success", 
                        parent_id=tool_event_id, 
                        data={
                            "result": result.content[:500],
                            "delta_applied": bool(getattr(result, "state_delta", None))
                        }
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_call.name}: {str(e)}"
                    context_messages.append(ToolResultMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=error_msg
                    ))
                    consecutive_failures += 1
                    yield AgentEvent(type="tool_execution_error", parent_id=tool_event_id, data={"error": error_msg})
            
            # 7. Steering Phase (Mid-Execution logging & non-blocking observation check)
            # If user injects message mid tool-calling, stop remaining tools.
            if hasattr(config, "get_steering_messages"):
                steering = await config.get_steering_messages()
                if steering:
                    context_messages.extend(steering)
                    steering_interrupted = True
                    yield AgentEvent(
                        type="steering_interruption", 
                        parent_id=turn_event_id, 
                        data={"steering_messages": steering}
                    )
                    break # Abort remaining tools in this batch and return to LLM reasoning loop
                    
        yield AgentEvent(
            type="turn_end", 
            parent_id=turn_event_id, 
            data={"message": assistant_msg, "tools_executed": len(assistant_msg.tool_calls), "interrupted": steering_interrupted}
        )
        
    yield AgentEvent(type="agent_end", parent_id=loop_event_id, data={"messages": context_messages})
