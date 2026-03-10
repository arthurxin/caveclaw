import asyncio
from typing import List, AsyncGenerator, Any, Optional, Callable
from .types import AgentState, AgentMessage, AgentTool, AgentLoopConfig, AgentEvent
from .agent_loop import run_loop


class Agent:
    """Facade class mirroring pi-agent-core/agent.ts. 
    Maintains session state and exposes high level control like `steer` and `follow_up` queues.
    """
    
    def __init__(self, config: AgentLoopConfig, tools: Optional[List[AgentTool]] = None):
        self.state = AgentState(tools=tools or [])
        self.config = config
        
        self._steering_queue: List[AgentMessage] = []
        self._followup_queue: List[AgentMessage] = []
        
        # Override config queue consumption 
        # By dynamically proxying the user's config hooks, the Agent class injects its own thread-safe mid-run UI queues.
        self._original_get_steering = getattr(config, "get_steering_messages", None)
        setattr(config, "get_steering_messages", self._get_steering_messages)
        
        self._original_get_followup = getattr(config, "get_followup_messages", None)
        setattr(config, "get_followup_messages", self._get_followup_messages)

    async def _get_steering_messages(self) -> List[AgentMessage]:
        msgs = self._steering_queue.copy()
        self._steering_queue.clear()
        
        if self._original_get_steering:
            extra = await self._original_get_steering()
            msgs.extend(extra)
        return msgs

    async def _get_followup_messages(self) -> List[AgentMessage]:
        msgs = self._followup_queue.copy()
        self._followup_queue.clear()
        
        if self._original_get_followup:
            extra = await self._original_get_followup()
            msgs.extend(extra)
        return msgs

    def set_system_prompt(self, prompt: str):
        self.state.system_prompt = prompt

    def replace_messages(self, messages: List[AgentMessage]):
        self.state.messages = messages

    def append_message(self, message: AgentMessage):
        self.state.messages.append(message)

    def steer(self, message: AgentMessage):
        """Queue a steering message to interrupt the agent mid-run.
        Delivered after the current tool execution, aborts remaining batch tools.
        """
        self._steering_queue.append(message)

    def follow_up(self, message: AgentMessage):
        """Queue a follow-up message to execute after the agent's turn completely stops."""
        self._followup_queue.append(message)

    def clear_queues(self):
        self._steering_queue.clear()
        self._followup_queue.clear()

    async def prompt(self, prompt: AgentMessage, stream_fn: Callable) -> AsyncGenerator[AgentEvent, None]:
        """Core execution hook. Given a prompt, spins up the event loop."""
        self.state.messages.append(prompt)
        self.state.is_streaming = True
        
        try:
            # Yield transparent UI events up to consumer app
            async for event in run_loop(self.state.messages, self.state.tools, self.config, stream_fn):
                yield event
        finally:
            self.state.is_streaming = False
