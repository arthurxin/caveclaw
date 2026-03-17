from openai import OpenAI
import re



class MinimaxCalling:
    """Minimax API 调用封装类，支持 tool calling 和 reasoning 解析"""
    
    def __init__(self, model_id, api_key, api_base, temperature=0.6,top_p=0.7,max_tokens=160000,stream=False):
        self.messages = []
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
    
    def get_completion(self, messages=None, tools=None):
        if messages:
            self.messages=messages
        params = {
            "model": self.model_id,
            "messages": self.messages,
            "extra_body": {"reasoning_split": True},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }
        
        # 只有当 tools 存在时才添加
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**params)
        
        # 打印 tokens 用量
        if response.usage:
            print(f"[Token Usage] prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
        
        message = response.choices[0].message
        
        # 获取 .content 字符串，不是 message 对象
        raw_content = message.content or ""
        
        # 提取推理内容
        think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None
        
        # 提取正文内容
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        # 把助手回复添加到消息历史
        self.messages.append(message)
        
        # 返回包含所有信息的字典
        return {
            "reasoning": reasoning,
            "content": content,
            "tool_calls": message.tool_calls,
            "raw_message": message
        }


if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    
    load_dotenv()

    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "getWeather",
                "description": "获得指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "getSum",
                "description": "计算数字的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "要相加的数字列表"
                        }
                    },
                    "required": ["numbers"]
                }
            }
        }
    ]
    
    # 模拟工具函数
    def get_weather(city: str):
        weather_data = {
            "北京": "晴天, 31°C",
            "上海": "多云, 32°C",
            "广州": "阵雨, 30°C",
        }
        return weather_data.get(city, f"{city}: 未知天气")
    
    def get_sum(numbers):
        return sum(numbers)
    
    def execute_tool_call(tool_call):
        """执行单个工具调用"""
        args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "getWeather":
            result = get_weather(args["city"])
        elif tool_call.function.name == "getSum":
            result = get_sum(args["numbers"])
        else:
            result = f"未知工具: {tool_call.function.name}"
        
        print(f"  > 调用 {tool_call.function.name}({args}) => {result}")
        return result
    
    # 初始化
    api_key = os.getenv("API_KEY")
    api_base = os.getenv("BASE_URL")
    model_id = os.getenv("MODEL_ID")
    client = MinimaxCalling(model_id, api_key, api_base)
    
    prompt = "北京和上海的天气怎么样？把这两个城市的温度加起来"
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以调用工具获取天气信息和计算数字。"},
        {"role": "user", "content": prompt}
    ]
    
    print(f"用户问题: {prompt}")
    print("=" * 50)
    
    # 第一次调用
    result = client.get_completion(messages, tools=tools)
    
    print(f"\n[推理过程]\n{result['reasoning'][:200] + '...' if result['reasoning'] and len(result['reasoning']) > 200 else result['reasoning']}")
    print(f"\n[回复内容] {result['content']}")
    print(f"[工具调用] {result['tool_calls']}")
    
    # 循环处理工具调用
    round_num = 1
    while result["tool_calls"]:
        print(f"\n{'=' * 50}")
        print(f"第 {round_num} 轮工具调用:")
        
        for tool_call in result["tool_calls"]:
            tool_result = execute_tool_call(tool_call)
            
            # 添加工具结果到消息历史（直接操作 client.messages）
            client.messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(tool_result)
            })
        
        # 继续调用模型（无需传 messages，类内部已维护）
        result = client.get_completion(tools=tools)
        round_num += 1
        
        print(f"\n[推理过程]\n{result['reasoning'][:200] + '...' if result['reasoning'] and len(result['reasoning']) > 200 else result['reasoning']}")
        print(f"\n[回复内容] {result['content']}")
        print(f"[工具调用] {result['tool_calls']}")
    
    print(f"\n{'=' * 50}")
    print("最终回复:")
    print(result["content"])