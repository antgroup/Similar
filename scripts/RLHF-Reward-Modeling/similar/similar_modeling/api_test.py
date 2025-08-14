import requests
import json

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 直接调用生产环境会报错
# url = "http://cv.gateway.alipay.com/ua/invoke"
url = "http://cv-cross.gateway.alipay.com/ua/invoke"

data = {
    "serviceCode": "Reward_Model_SFT_Qwen_VL_Chat",
    "uri": "SFT_Qwen_VL_Chat_v1",
    "attributes": {
        "_TIMEOUT_": "30000",
        "_ROUTE_": "MAYA",
        # "_APP_TOKEN_": "your-app-token"
    },
    "params": {
        "features": {
            "text": "勒布朗詹姆斯有几个NBA总冠军？请直接回复数字。"
        }
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)