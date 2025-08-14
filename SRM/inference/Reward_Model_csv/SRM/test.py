import json

from utils import (
    get_default_config,
    ask_chatgpt
)

# 定义需要调用的函数列表（此处以获取天气为例）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定地区的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或区县，例如：北京市朝阳区",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        }
    }
]


# 模拟真实函数（实际开发中应替换为真实API调用）
def get_current_weather(location, unit="celsius"):
    """模拟天气数据查询"""
    return f"{location}当前天气为25度，{unit}，晴间多云"


def ask_chatgpt_ant(messages):
    param = get_default_config(model="gpt-4o")
    param["queryConditions"]["model"] = "gpt-4o"
    param["queryConditions"]["temperature"] = "0.2"

    param["queryConditions"]["messages"] = messages
    param["queryConditions"]["tools"] = tools
    param["queryConditions"]["tool_choice"] = "auto"
    try:
        response = ask_chatgpt(param)
        # print(response)
        return response
    except Exception as e:
        print("error: ", e)
        return False


messages = [
    {
        "role":"user",
        "content":"北京现在天气怎么样？"
    }
]


while True:
    response = ask_chatgpt_ant(messages)
    if (response != False):
        print(response)
        print("\n")
        # 检查是否需要调用函数
        if response['tool_calls']:
            # 遍历所有需要调用的函数
            for tool_call in response['tool_calls']:
                print("tool_call: ", tool_call)
                function_name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])

                # 调用对应函数
                if function_name == "get_current_weather":
                    weather_info = get_current_weather(
                        location=arguments.get("location"),
                        unit=arguments.get("unit", "celsius")
                    )
                    print("函数调用结果:", weather_info)
        else:
            print("模型直接回复:", response['content'])
        break