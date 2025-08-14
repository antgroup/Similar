import os
import json
import csv
import requests

url = "https://api.superbed.cn/upload"

# 通过链接上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057", "src": "https://ww1.sinaimg.cn/large/005YhI8igy1fv09liyz9nj30qo0hsn0e"})

# 通过文件上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open("demo.jpg", "rb")})
# print(resp.json())

target_path = "generate_data/9.29_csv/tasks/webarena/22-1/data.csv"

# if not os.path.isdir(target_path):  # 如果没有该文件则新建一个
#     os.makedirs(target_path)

filepath = "generate_data/9.27/tasks/webarena/22-1"
task_filepath = "tasks/webarena/22.json"
result_filepath = "generate_data/9.27/tasks/webarena/22-1"
json_filepath = "generate_data/9.27/tasks/webarena/22-1/evaluation_score.json"

with open(task_filepath, 'r') as f:
    content = json.load(f)

instruction = content["intent"]
print("instruction = ", instruction)

with open(target_path, 'w', newline='') as file:
    # csv.writer(file)
    fields = ['instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    with open(json_filepath, 'r', encoding='UTF-8') as f:
        json_now = json.load(f)

        for (key, value) in json_now.items():

            img = filepath + '/webpage' + str(key) + '.png'
            resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open(img, "rb")})
            # print(resp.json())
            observation_url = resp.json()['url']
            print("observation_url = ", observation_url)

            writer.writerow({'instruction': instruction, 'step_idx': int(value['step_idx']), 'observation_url': observation_url,
                             'action': str(value['action']), 'IP': float(value['IP']), 'E': float(value['E']), 'TC': float(value['TC']), 'TR': float(value['TR']), 'C': float(value['C'])})

