import os
import json
import csv
import requests
import shutil

def get_action(action_list):
    # print("action_list = ", action_list)
    if len(action_list) == 0:
        return 'hover'

    res = ''
    for action in action_list:
        if not 'import' in action:
            res += action + "\n"
            continue

        now_action_list = action.split("\n")
        # print(now_action_list)
        for now_action in now_action_list:
            # print(len(now_action))
            if (len(now_action) < 1):
                continue
            # if (now_action[0] == '#'):
            #     res += now_action[2:] + ' \n'
            if '#' in now_action:
                res += now_action[now_action.find('#') + 1:] + ' \n'
                # print(now_action)
        # print("\n")
    # print("action = ", res)
    return res


url = "https://api.superbed.cn/upload"

# 通过链接上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057", "src": "https://ww1.sinaimg.cn/large/005YhI8igy1fv09liyz9nj30qo0hsn0e"})

# 通过文件上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open("demo.jpg", "rb")})
# print(resp.json())

filepath = "generate_data/" # 标注数据文件夹
task_filepath = "evaluation_examples/examples/Windows" # task文件夹

num_success = 0
num_useless = 0
num_already = 0
num_empty = 0
num_no = 0
num_have = 0
num_lack = 0
# delete_list = []
lack_list = []
empty_list = []
error_list = []

dict =  {'0810415c-bde4-4443-9047-d5f70165a697': 1, '09a37c51-e625-49f4-a514-20a773797a8a': 2, '0b17a146-2934-46c7-8727-73ff6b6483e8': 3, '0e763496-b6bb-4508-a427-fad0b6c3e195': 4, '185f29bd-5da0-40a6-b69c-ba7f4e0324ef': 5, '1f18aa87-af6f-41ef-9853-cdb8f32ebdea': 6, '26150609-0da3-4a7d-8868-0faf9c5f01bb': 7, '26660ad1-6ebb-4f59-8cba-a8432dfe8d38': 8, '2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e': 9, '3a93cae4-ad3e-403e-8c12-65303b271818': 10, '3aaa4e37-dc91-482e-99af-132a612d40f3': 11, '3b27600c-3668-4abd-8f84-7bcdebbccbdb': 12, '3ef2b351-8a84-4ff2-8724-d86eae9b842e': 13, '4188d3a4-077d-46b7-9c86-23e1a036f6c1': 14, '455d3c66-7dc6-4537-a39a-36d3e9119df7': 15, '46407397-a7d5-4c6b-92c6-dbe038b1457b': 16, '4bcb1253-a636-4df4-8cb0-a35c04dfef31': 17, '51b11269-2ca8-4b2a-9163-f21758420e78': 18, '550ce7e7-747b-495f-b122-acdc4d0b8e54': 19, '5d901039-a89c-4bfb-967b-bf66f4df075e': 20, '6054afcb-5bab-4702-90a0-b259b5d3217c': 21, '6d72aad6-187a-4392-a4c4-ed87269c51cf': 22, '6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a': 23, '6f81754e-285d-4ce0-b59e-af7edb02d108': 24, '74d5859f-ed66-4d3e-aa0e-93d7a592ce41': 25, '7a4e4bc8-922c-4c84-865c-25ba34136be1': 26, '7efeb4b1-3d19-4762-b163-63328d66303b': 27, '897e3b53-5d4d-444b-85cb-2cdc8a97d903': 28, '8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14': 29, '8e116af7-7db7-4e35-a68b-b0939c066c78': 30, '9ec204e4-f0a3-42f8-8458-b772a6797cab': 31, 'a097acff-6266-4291-9fbd-137af7ecd439': 32, 'a82b78bb-7fde-4cb3-94a4-035baf10bcf0': 33, 'a9f325aa-8c05-4e4f-8341-9e4358565f4f': 34, 'abed40dc-063f-4598-8ba5-9fe749c0615d': 35, 'b21acd93-60fd-4127-8a43-2f5178f4a830': 36, 'b5062e3e-641c-4e3a-907b-ac864d2e7652': 37, 'c867c42d-a52d-4a24-8ae3-f75d256b5618': 38, 'ce88f674-ab7a-43da-9201-468d38539e4a': 39, 'd1acdb87-bb67-4f30-84aa-990e56a09c92': 40, 'da52d699-e8d2-4dc5-9191-a2199e0b6a9b': 41, 'deec51c9-3b1e-4b9e-993c-4776f20e8bb2': 42, 'e2392362-125e-4f76-a2ee-524b183a3412': 43, 'e528b65e-1107-4b8c-8988-490e4fece599': 44, 'eb03d19a-b88d-4de4-8a64-ca0ac66f426b': 45, 'eb303e01-261e-4972-8c07-c9b4e7a4922a': 46, 'ecb0df7a-4e8d-4a03-b162-053391d3afaf': 47, 'ecc2413d-8a48-416e-a3a2-d30106ca36cb': 48, 'ecc2413d-8a48-416e-a3a2-d30106ca36cb-': 49, 'f918266a-b3e0-4914-865d-4faa564f1aef': 50}
dict_idx = 0

for filedir in os.listdir(filepath):
    # print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if filedir[-2] == '-':
        task_id = filedir[:-4]
    else:
        task_id = filedir

    # print("task_id = ", task_id)

    instruction = ''
    for filedir2 in os.listdir(task_filepath):
        flag = 0
        if not os.path.isdir(os.path.join(task_filepath, filedir2)):
            continue
        for filrdir3 in os.listdir(os.path.join(task_filepath, filedir2)):
            if filrdir3[:-5] == task_id:
                # 读取任务的instruction
                task_file = os.path.join(task_filepath, filedir2, task_id + ".json") # 具体的任务文件

                with open(task_file, 'r') as f:
                    task = json.load(f)

                instruction = task["instruction"]

                flag = 1
                break
        if flag:
            break

    # print("instruction = ", instruction)
    if instruction == '':
        print("without instruction")
        print("\nnow = ", filedir)
        continue

    idx = 0
    if task_id not in dict:
        dict_idx = dict_idx + 1
        dict[task_id] = dict_idx
        idx = dict_idx
    else:
        idx = dict[task_id]

    # print("idx = ", idx)

    # 存入到csv文件中
    # csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + filedir[-4:] + ".csv"
    if filedir[-2] == '-':
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + filedir[-4:] + ".csv"
    else:
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + ".csv"
    # print("csv_file = ", csv_file)

    if os.path.exists(csv_file):
        # print("csv_file = ", csv_file)
        if len(open(csv_file).readlines()) <= 1:
            num_useless += 1
            os.remove(csv_file)
            # continue

    if os.path.exists(csv_file):
        num_already += 1
        continue


    evaluation_score_file = os.path.join(filepath, filedir, "evaluation_score.json") # 步骤级多维度评分文件
    if os.path.exists(evaluation_score_file): # 该文件数据有效

        print("\nnow = ", filedir)
        print("csv_file = ", csv_file)
        print("\n")

        if os.path.exists(csv_file):
            # num_already += 1
            continue

        with open(csv_file, 'w', newline='') as file:
            fields = ['instruction', 'step_idx', 'observation_url', 'action', 'action_origin', 'IP', 'E', 'TC', 'TR', 'C']
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

                action = get_action(value['action'])
                if action == '':
                    print("action is None")
                    break

                try:
                    resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open(img, "rb")})
                    print("resp.json() = ", resp.json())
                    observation_url = resp.json()['url']

                    writer.writerow(
                        {'instruction': instruction, 'step_idx': int(value['step_idx']), 'observation_url': observation_url,
                         'action': action, 'action_origin': str(value['action']) ,'IP': float(value['IP']), 'E': float(value['E']),
                         'TC': float(value['TC']), 'TR': float(value['TR']), 'C': float(value['C'])})
                except Exception as e:
                    continue
                    # break
            num_success += 1

            # if (num_success == 1):
            #     break

            # if pic_num:
            #     num_have += 1
            #     print("\nthere are images")
            #
            #     i = 1
            #     for (key, value) in evaluation_score_dict.items():
            #         if int(key) != i:
            #             error_list.append(filedir)
            #             break
            #         # try:
            #         img_ori = os.path.join(filepath, str(idx)) + '/webpage' + str(i) + '.png'
            #         img = os.path.join(filepath, filedir) + '/webpage' + str(i) + '.png'
            #         if not os.path.exists(img):
            #             num_lack += 1
            #             lack_list.append(filedir)
            #             break
            #             shutil.copyfile(img_ori, img)
            #         i += 1
            # else:
            #     num_no += 1
            #     print("\nthere no images")
            #     empty_list.append(filedir)
            #     path = os.path.join(filepath, filedir)
            #     shutil.rmtree(path)
    else: # 该文件夹数据无效
        num_empty += 1
        continue

# print("\ndict = ", dict)
# print("\n")
print("num_already = ", num_already)
print("num_empty = ", num_empty)
print("num_have = ", num_have)
print("num_lack = ", num_lack)
print("num_no = ", num_no)
print("num_useless = ", num_useless)

# print("lack_list = ", lack_list)
# print("num lack_list = ", len(lack_list))
# print("\n")
#
# print("empty_list = ", empty_list)
# print("num empty_list = ", len(empty_list))
# print("\n")
#
# print("error_list = ", error_list)
# print("num error_list = ", len(error_list))
# print("\n")

