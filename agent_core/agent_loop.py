import asyncio
from typing import AsyncGenerator, List, Callable, Optional, Dict, Any, Tuple

from .types import (
    AgentMessage, AgentLoopConfig, AgentTool, AgentEvent,
    Message, AssistantMessage, ToolCall, ToolResultMessage
)

async def _stream_assistant_response(
    messages: List[Message],
    stream_fn: Callable[..., AsyncGenerator[Any, None]],
    config: AgentLoopConfig
) -> AssistantMessage:
    """Wrapper that reads from the provided stream_fn and constructs standard AssistantMessage."""
    assistant_msg = AssistantMessage(content="")
    
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


async def execute_tool_calls(
    tools: List[AgentTool],
    assistant_message: AssistantMessage,
    get_steering_messages: Optional[Callable[[], List[AgentMessage]]]
) -> Tuple[List[ToolResultMessage], List[AgentMessage]]:
    """Execute matched tools sequentially and check for steering messages mid-execution."""
    tool_results = []
    
    for tool_call in assistant_message.tool_calls or []:
        tool = next((t for t in tools if t.name == tool_call.name), None)
        
        if not tool:
            tool_results.append(ToolResultMessage(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Tool {tool_call.name} not found."
            ))
            continue
            
        try:
            # Execute tool and append result
            result = await tool.execute(tool_call.id, tool_call.arguments)
            tool_results.append(ToolResultMessage(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=result.content
            ))
        except Exception as e:
            tool_results.append(ToolResultMessage(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing tool {tool_call.name}: {str(e)}"
            ))
            
        # 动态干预检查 (Steering Phase) - If user injects message mid tool-calling, stop remaining tools.
        if get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                return tool_results, steering
                
    return tool_results, []


async def run_loop(
    context_messages: List[AgentMessage],
    tools: List[AgentTool],
    config: AgentLoopConfig,
    stream_fn: Callable[..., AsyncGenerator[Any, None]]
) -> AsyncGenerator[AgentEvent, None]:
    """The core state machine loop (equivalent to pi-agent-core/agent-loop.ts)."""
    
    yield AgentEvent(type="agent_start")
    
    while True:
        # 1. Transform Context (e.g. context window pruning)
        if hasattr(config, "transform_context"):
            context_messages = await config.transform_context(context_messages)
            
        # 2. Convert to LLM format (filter out CustomMessages/UI elements)
        llm_messages = await config.convert_to_llm(context_messages)
        
        yield AgentEvent(type="turn_start")
        
        # 3. Stream Response
        assistant_msg = await _stream_assistant_response(llm_messages, stream_fn, config)
        context_messages.append(assistant_msg)
        
        # 4. Check array of Tool Calls
        if not assistant_msg.tool_calls:
            # End of turn, no tools
            yield AgentEvent(type="turn_end", data={"message": assistant_msg})
            
            # 5. Check Follow Up messages queue
            if hasattr(config, "get_followup_messages"):
                followup = await config.get_followup_messages()
                if followup:
                    context_messages.extend(followup)
                    continue
            break # Fully finished
            
        # 6. Execute Tools
        tool_results, steering_msgs = await execute_tool_calls(
            tools, assistant_msg, 
            getattr(config, "get_steering_messages", None)
        )
        context_messages.extend(tool_results)
        
        # If interrupted by steering, injection starts a new loop iteration instantly.
        if steering_msgs:
            context_messages.extend(steering_msgs)
            
        yield AgentEvent(type="turn_end", data={"message": assistant_msg, "tool_results": tool_results})
        
    yield AgentEvent(type="agent_end", data={"messages": context_messages})
