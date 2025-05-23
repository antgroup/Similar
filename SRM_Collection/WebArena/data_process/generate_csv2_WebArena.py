import os
import json
import csv
import requests
import shutil

url = "https://api.superbed.cn/upload"

# 通过链接上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057", "src": "https://ww1.sinaimg.cn/large/005YhI8igy1fv09liyz9nj30qo0hsn0e"})

# 通过文件上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open("demo.jpg", "rb")})
# print(resp.json())

filepath = "generate_data/10.8/tasks/webarena" # 标注数据文件夹
task_filepath = "tasks/webarena" # task文件夹

num_already = 0
num_empty = 0
num_no = 0
num_have = 0
num_lack = 0
delete_list = []

for filedir in os.listdir(filepath):
    # print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if '-' in filedir:
        idx = filedir[:filedir.find('-')]
    else:
        idx = filedir

    # print("idx = ", idx)

    # 读取任务的instruction
    task_file = os.path.join(task_filepath, idx + ".json") # 具体的任务文件

    with open(task_file, 'r') as f:
        task = json.load(f)

    instruction = task["intent"]
    # print("instruction = ", instruction)

    # 存入到csv文件中
    csv_file = os.path.join(filepath, filedir) + "/" + filedir + ".csv"
    # print("csv_file = ", csv_file)

    if os.path.exists(csv_file):
        if len(open(csv_file).readlines()) <= 1:
            os.remove(csv_file)

    if os.path.exists(csv_file):
        num_already += 1
        continue


    evaluation_score_file = os.path.join(filepath, filedir, "evaluation_score.json") # 步骤级多维度评分文件
    if os.path.exists(evaluation_score_file): # 该文件数据有效

        print("\nnow = ", filedir)
        print("csv_file = ", csv_file)

        with open(csv_file, 'w', newline='') as file:
            fields = ['instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            with open(evaluation_score_file, 'r', encoding='UTF-8') as f:
                evaluation_score_dict = json.load(f)

            pic_num = 0
            for (key, value) in evaluation_score_dict.items():
                img = os.path.join(filepath, filedir) + '/webpage' + str(key) + '.png'
                if os.path.exists(img):
                    pic_num = 1
                    # continue
                else:
                    flag = 1
                    # continue

                try:
                    resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open(img, "rb")})
                    print("resp.json() = ", resp.json())
                    observation_url = resp.json()['url']

                    writer.writerow(
                        {'instruction': instruction, 'step_idx': int(value['step_idx']), 'observation_url': observation_url,
                         'action': str(value['action']), 'IP': float(value['IP']), 'E': float(value['E']),
                         'TC': float(value['TC']), 'TR': float(value['TR']), 'C': float(value['C'])})
                except Exception as e:
                    continue
                    # break
            if pic_num:
                num_have += 1
                print("\nthere are images")

                i = 1
                for (key, value) in evaluation_score_dict.items():
                    # try:
                    img_ori = os.path.join(filepath, idx) + '/webpage' + str(key) + '.png'
                    img = os.path.join(filepath, filedir) + '/webpage' + str(i) + '.png'
                    if not os.path.exists(img):
                        num_lack += 1
                        break
                        # shutil.copyfile(img_ori, img)
                    i += 1
            else:
                num_no += 1
                print("\nthere no images")
                # path = os.path.join(filepath, filedir)
                # shutil.rmtree(path)
    else: # 该文件夹数据无效
        num_empty += 1
        continue

print("num_already = ", num_already)
print("num_empty = ", num_empty)
print("num_have = ", num_have)
print("num_lack = ", num_lack)
print("num_no = ", num_no)

