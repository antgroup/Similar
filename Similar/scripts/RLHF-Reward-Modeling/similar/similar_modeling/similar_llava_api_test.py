import requests
import json

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 直接调用生产环境会报错
# url = "http://cv.gateway.alipay.com/ua/invoke"
url = "http://cv-cross.gateway.alipay.com/ua/invoke"

data = {
    "serviceCode": "Reward_Model_ArmoRM",
    "uri": "ArmoRM_Llama3_8B_v02",
    "attributes": {
        "_TIMEOUT_": "30000",
        "_ROUTE_": "MAYA",
        # "_APP_TOKEN_": "your-app-token"
    },
    "params": {
        "features": {
            "type_now": "IP",
            "instruction": "In Simple Calendar Pro, create a recurring calendar event titled 'Review session for Budget Planning' starting on 2023-10-15 at 14h. The event recurs weekly, forever, and lasts for 60 minutes each occurrence. The event description should be 'We will understand business objectives. Remember to confirm attendance.'.",
            "reason_steps": "Step 1 :  open_app Simple Calendar Pro  Step 2 : click New Event  Step 3 : click Event  Step 4 :  input_text Review session for Budget Planning  Step 5 : click 16:00  Step 6 : click 14 hours  Step 7 : click OK  Step 8 : click No repetition  Step 9 : click Weekly  Step 10 :  input_text We will understand business objectives. Remember to confirm attendance.",
            "observation_url": "https://pic.imgdb.cn/item/6719965ed29ded1a8c609d69.jpg",
            "action": "click 14:00"
        }
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)