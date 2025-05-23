import requests
import json

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 直接调用生产环境会报错
# url = "http://cv.gateway.alipay.com/ua/invoke"
url = "http://cv-cross.gateway.alipay.com/ua/invoke"

data = {
    "serviceCode": "DeepSeek_R1_Test_Agent",
    "uri": "DeepSeek_R1_Test_Agent_v1",
    "attributes": {
        "_TIMEOUT_": "30000",
        "_ROUTE_": "MAYA",
        # "_APP_TOKEN_": "your-app-token"
    },
    "params": {
        "features": {
            "data": "{ \t\"__entry_point__\": \"openai.chat.completion\", \t\"model\":\"auto\", \t\"temperature\": 0.2, \t\"top_p\": 1, \t\"messages\": [ \t\t{ \t\t\t\"role\": \"system\", \t\t\t\"content\": \"You are a helpful assistant.\" \t\t}, \t\t{ \t\t\t\"role\": \"user\", \t\t\t\"content\": \"讲个笑话\" \t\t} \t], \t\"stream\": false }"
        }
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)