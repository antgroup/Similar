import json
import requests

# 预发
url = "https://paiplusinferencepre.alipay.com/inference/2327626b2f91dd5a_semantic_matching/v1"
# 线上
# url = "https://paiplusinference.alipay.com/inference/2327626b2f91dd5a_semantic_matching/v1"

head = {
    "Content-Type": "application/json",
    "MPS-app-name": "test",
    "MPS-trace-id": "trace_lx",
    "MPS-http-version": "1.0"
}
prompt = {
    "features": {
        "query":"你是谁",
        "choices":'["门诊", "住院", "第一个", "第二个", "听不懂"]'
    }
}

response = requests.post(
    url=url,
    headers=head,
    data=json.dumps(prompt)
)

print(response.text)