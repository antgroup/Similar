import os
import requests

url = "https://api.superbed.cn/upload"

# 通过链接上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057", "src": "https://ww1.sinaimg.cn/large/005YhI8igy1fv09liyz9nj30qo0hsn0e"})

# 通过文件上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open("demo.jpg", "rb")})

filepath = "generate_data/9.27/tasks/webarena/22-1/"

for file in os.listdir(filepath):
    print("file = ", file)
    if file.endswith(".png"):
        resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open(filepath + file, "rb")})
        print(resp.json())
