"""
AgentLoop - 通用的 LLM + 工具调用循环

提供递归式的 agent 循环，支持：
- 多轮工具调用
- 状态管理 (loop_response)
- 递归审视
- 可选的保存回调
"""

import json
from typing import List, Callable, Any, Optional
from minimax_calling import MinimaxCalling


class AgentLoop:
    """通用的 LLM + 工具调用循环"""
    
    def __init__(
        self,
        client: MinimaxCalling,
        tools: List[dict],
        tool_executor: Callable[[Any, dict], str],
        max_recursion: int = 10,
        max_rounds: int = 20
    ):
        """
        初始化 AgentLoop
        
        Args:
            client: MinimaxCalling 客户端
            tools: 工具定义列表
            tool_executor: 工具执行函数，签名为 (tool_call, loop_response) -> str
            max_recursion: 最大递归次数
            max_rounds: 单次递归内最大对话轮数
        """
        self.client = client
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_recursion = max_recursion
        self.max_rounds = max_rounds
    
    def run(
        self,
        system_prompt: str,
        build_user_message: Callable[[dict], str],
        loop_response: dict = None,
        on_recursion_end: Callable[[dict, int], None] = None,
        verbose: bool = True
    ) -> dict:
        """
        运行 agent 循环
        
        Args:
            system_prompt: 系统提示词
            build_user_message: 构建用户消息的函数，签名为 (loop_response) -> str
            loop_response: 初始状态字典，会被工具执行器修改
            on_recursion_end: 每次递归结束时的回调，签名为 (loop_response, recursion_count) -> None
            verbose: 是否打印详细日志
            
        Returns:
            最终的 loop_response
        """
        if loop_response is None:
            loop_response = {}
        
        return self._recursive_run(
            system_prompt=system_prompt,
            build_user_message=build_user_message,
            loop_response=loop_response,
            on_recursion_end=on_recursion_end,
            verbose=verbose,
            recursion_count=0
        )
    
    def _recursive_run(
        self,
        system_prompt: str,
        build_user_message: Callable[[dict], str],
        loop_response: dict,
        on_recursion_end: Optional[Callable],
        verbose: bool,
        recursion_count: int
    ) -> dict:
        """递归执行"""
        if verbose:
            print(f"\n{'='*50}")
            print(f"递归第 {recursion_count + 1} 次 (最大 {self.max_recursion} 次)")
            print(f"{'='*50}")
        
        # 构建用户消息
        user_message = build_user_message(loop_response)
        
        # 重置消息历史
        self.client.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 内部循环
        round_count = 0
        first_round_has_tools = False
        
        while round_count < self.max_rounds:
            round_count += 1
            if verbose:
                print(f"\n  --- 第 {round_count} 轮对话 ---")
            
            # 调用 LLM
            result = self.client.get_completion(tools=self.tools)
            
            if verbose:
                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"[回复] {content_preview}")
                print(f"[工具] {len(result['tool_calls']) if result['tool_calls'] else 0} 个调用")
            
            # 如果没有工具调用
            if not result["tool_calls"]:
                if verbose:
                    print("  本轮无工具调用")
                if round_count == 1:
                    first_round_has_tools = False
                break
            
            # 记录第一轮有工具调用
            if round_count == 1:
                first_round_has_tools = True
            
            # 添加 assistant 消息到历史
            assistant_message = {"role": "assistant", "content": result["content"]}
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in result["tool_calls"]
            ]
            self.client.messages.append(assistant_message)
            
            # 执行所有工具调用
            for tool_call in result["tool_calls"]:
                if verbose:
                    args_preview = tool_call.function.arguments[:100] + "..." if len(tool_call.function.arguments) > 100 else tool_call.function.arguments
                    print(f"  执行: {tool_call.function.name}({args_preview})")
                
                tool_result = self.tool_executor(tool_call, loop_response)
                
                if verbose:
                    print(f"  结果: {tool_result}")
                
                # 添加工具结果到消息历史
                self.client.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_result
                })
        
        if verbose:
            print(f"\n  递归 {recursion_count + 1} 完成，共 {round_count} 轮对话")
        
        # 调用回调
        if on_recursion_end:
            on_recursion_end(loop_response, recursion_count)
        
        # 判断是否进入下一次递归
        if first_round_has_tools and recursion_count + 1 < self.max_recursion:
            if verbose:
                print("  进入下一次递归...")
            return self._recursive_run(
                system_prompt=system_prompt,
                build_user_message=build_user_message,
                loop_response=loop_response,
                on_recursion_end=on_recursion_end,
                verbose=verbose,
                recursion_count=recursion_count + 1
            )
        elif not first_round_has_tools:
            if verbose:
                print("  第一轮无工具调用，模型认为已完成")
        else:
            if verbose:
                print(f"  达到最大递归次数 ({self.max_recursion})")
        
        return loop_response


# ========== 通用编辑工具 ==========

def create_edit_tools(data_key: str, item_name: str = "项目") -> List[dict]:
    """
    创建通用的编辑工具定义
    
    Args:
        data_key: loop_response 中存储数据的键名，如 "scenes", "characters"
        item_name: 项目名称，用于描述，如 "场景", "角色"
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "append",
                "description": f"批量添加{item_name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": f"要添加的{item_name}列表",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "data": {"type": "object"}
                                },
                                "required": ["key", "data"]
                            }
                        }
                    },
                    "required": ["items"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update",
                "description": f"批量更新{item_name}的字段",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "field": {"type": "string"},
                                    "value": {}
                                },
                                "required": ["key", "field", "value"]
                            }
                        }
                    },
                    "required": ["updates"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete",
                "description": f"批量删除{item_name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["keys"]
                }
            }
        }
    ]


def create_edit_executor(data_key: str):
    """
    创建通用的编辑工具执行器
    
    Args:
        data_key: loop_response 中存储数据的键名
    """
    def executor(tool_call, loop_response) -> str:
        try:
            args = json.loads(tool_call.function.arguments)
            name = tool_call.function.name
            
            # 确保数据键存在
            if data_key not in loop_response:
                loop_response[data_key] = {}
            
            data = loop_response[data_key]
            results = []
            
            if name == "append":
                for item in args["items"]:
                    key, item_data = item["key"], item["data"]
                    if key in data:
                        results.append(f"错误: {key} 已存在")
                    else:
                        data[key] = item_data
                        results.append(f"已添加 {key}")
            
            elif name == "update":
                for item in args["updates"]:
                    key, field, value = item["key"], item["field"], item["value"]
                    if key not in data:
                        results.append(f"错误: {key} 不存在")
                    else:
                        data[key][field] = value
                        results.append(f"已更新 {key}.{field}")
            
            elif name == "delete":
                for key in args["keys"]:
                    if key not in data:
                        results.append(f"错误: {key} 不存在")
                    else:
                        del data[key]
                        results.append(f"已删除 {key}")
            
            else:
                return f"未知工具: {name}"
            
            return "; ".join(results)
        
        except Exception as e:
            return f"执行失败: {str(e)}"
    
    return executor


# ========== 测试用例 ==========

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("BASE_URL")
    model_id = os.getenv("MODEL_ID")

    # 创建客户端
    client = MinimaxCalling(
        model_id=model_id,
        api_key=api_key,
        api_base=api_base
    )
    
    # 创建工具
    tools = create_edit_tools("items", "待办事项")
    executor = create_edit_executor("items")
    
    # 创建 AgentLoop
    agent = AgentLoop(
        client=client,
        tools=tools,
        tool_executor=executor,
        max_recursion=3,
        max_rounds=10
    )
    
    # 系统提示
    system_prompt = """你是一个待办事项管理助手。
    
你有三个工具可以修改待办事项：
1. append: 批量添加待办事项
2. update: 批量更新待办事项
3. delete: 批量删除待办事项

当你认为任务完成，不需要修改时，不要调用任何工具。"""
    
    # 构建用户消息
    def build_message(loop_response):
        items = loop_response.get("items", {})
        if items:
            items_str = json.dumps(items, ensure_ascii=False, indent=2)
            return f"当前待办事项:\n```json\n{items_str}\n```\n\n请检查是否完善。"
        else:
            return "请帮我创建三个待办事项：1.买菜 2.写代码 3.运动"
    
    # 运行
    result = agent.run(
        system_prompt=system_prompt,
        build_user_message=build_message,
        loop_response={"items": {}},
        verbose=True
    )
    
    print("\n\n========== 最终结果 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
