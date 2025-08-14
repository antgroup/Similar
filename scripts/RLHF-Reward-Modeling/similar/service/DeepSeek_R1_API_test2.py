import requests

url = "https://paiplusinference.alipay.com/inference/c668d5aef425295b_DeepSeek_R1_Test_Agent/DeepSeek_R1_Test_Agent_v1"
body = {"features":{},"tensorFeatures":{"data":{"shapes":[1],"stringValues":["{ \t\"__entry_point__\": \"openai.chat.completion\", \t\"model\":\"auto\", \t\"temperature\": 0.2, \t\"top_p\": 1, \t\"messages\": [ \t\t{ \t\t\t\"role\": \"system\", \t\t\t\"content\": \"You are a helpful assistant.\" \t\t}, \t\t{ \t\t\t\"role\": \"user\", \t\t\t\"content\": \"讲个笑话\" \t\t} \t], \t\"stream\": false }"]}}}
headers = {
    "Content-Type": "application/json;charset=utf-8",
    "MPS-app-name": "your-app-name",
    "MPS-http-version": "1.0",
    "MPS-trace-id": "your-trace-id"
}

r = requests.post(url=url, json=body, headers=headers)
res = r.json()
print(res)