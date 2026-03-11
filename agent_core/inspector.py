from typing import Any, Dict, List, Optional
import sys

from .types import EnvironmentInspector, AgentContext, StateReducer


class DataFrameReducer(StateReducer):
    """
    Safely reduces a Pandas DataFrame to avoid token explosion.
    Only captures shape, columns, bounding memory, and the top 3 rows.
    """
    def reduce(self, obj: Any) -> str:
        # Check if pandas is available without enforcing it as a strict dependency
        if 'pandas' in sys.modules and str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
            try:
                shape = obj.shape
                cols = list(obj.columns)
                head_str = obj.head(3).to_string()
                memory_usage = obj.memory_usage(deep=True).sum() / (1024 * 1024)
                
                return (
                    f"DataFrame:\n"
                    f"  - Shape: {shape[0]} rows x {shape[1]} cols\n"
                    f"  - Memory: ~{memory_usage:.2f} MB\n"
                    f"  - Columns: {cols}\n"
                    f"  - Head (3 rows):\n{head_str}"
                )
            except Exception as e:
                return f"[Error reducing DataFrame: {str(e)}]"
        return str(obj)


class ListReducer(StateReducer):
    """
    Reduces long lists to prevent spamming the LLM context.
    Displays the first few and last few items.
    """
    def __init__(self, max_items: int = 5):
        self.max_items = max_items
        
    def reduce(self, obj: Any) -> str:
        if isinstance(obj, list):
            if len(obj) <= self.max_items:
                return str(obj)
            else:
                head = obj[:self.max_items]
                return f"List (len={len(obj)}): {str(head)[:-1]}, ... <{len(obj) - self.max_items} more items>]"
        return str(obj)


class PythonRuntimeInspector(EnvironmentInspector):
    """
    In-Process Inspector that safely extracts the shared_memory state for the LLM.
    Uses registered reducers to prevent massive objects from breaking the context window.
    """
    def __init__(self, reducers: Optional[List[StateReducer]] = None):
        if reducers is None:
            self.reducers = [DataFrameReducer(), ListReducer()]
        else:
            self.reducers = reducers
            
    async def capture_state(self, context: AgentContext) -> str:
        """
        Takes a snapshot of the current `shared_memory`.
        Applies reducers recursively if needed (for simplicity, applied top-level here).
        """
        memory = context.shared_memory
        if not memory:
            return "Environment State: [Memory empty or unchanged]"
            
        summary_lines = ["--- Python Runtime State Feedback ---"]
        
        for key, val in memory.items():
            reduced_val = None
            
            # Find the first reducer that successfully alters the output format,
            # or just fallback to string casting.
            for reducer in self.reducers:
                reduced_attempt = reducer.reduce(val)
                # If the reducer actually did something instead of returning raw string
                if reduced_attempt != str(val) or type(reducer).__name__ == "DataFrameReducer" and "DataFrame:" in reduced_attempt:
                    reduced_val = reduced_attempt
                    break
                    
            if reduced_val is None:
                # Fallback for dicts and primitives
                if isinstance(val, dict):
                    keys = list(val.keys())
                    if len(keys) > 10:
                        reduced_val = f"Dict with {len(keys)} keys: {keys[:10]}..."
                    else:
                        reduced_val = str(val)
                else:
                    raw_str = str(val)
                    if len(raw_str) > 500:
                        reduced_val = raw_str[:500] + f"... <Truncated {len(raw_str) - 500} chars>"
                    else:
                        reduced_val = raw_str
                        
            summary_lines.append(f"[{key}]:\n{reduced_val}\n")
            
        return "\n".join(summary_lines)
