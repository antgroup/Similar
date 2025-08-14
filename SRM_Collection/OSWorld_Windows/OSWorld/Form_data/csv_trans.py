import os
import csv
import pandas as pd

filepath = "generate_data" # 标注数据文件夹

final_csv_file = filepath + "/" + "Reward_Model.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        # fields = ['task_ID', 'sub_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)
# print("final_data = \n", final_data)

dict =  {'0810415c-bde4-4443-9047-d5f70165a697': 1, '09a37c51-e625-49f4-a514-20a773797a8a': 2, '0b17a146-2934-46c7-8727-73ff6b6483e8': 3, '0e763496-b6bb-4508-a427-fad0b6c3e195': 4, '185f29bd-5da0-40a6-b69c-ba7f4e0324ef': 5, '1f18aa87-af6f-41ef-9853-cdb8f32ebdea': 6, '26150609-0da3-4a7d-8868-0faf9c5f01bb': 7, '26660ad1-6ebb-4f59-8cba-a8432dfe8d38': 8, '2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e': 9, '3a93cae4-ad3e-403e-8c12-65303b271818': 10, '3aaa4e37-dc91-482e-99af-132a612d40f3': 11, '3b27600c-3668-4abd-8f84-7bcdebbccbdb': 12, '3ef2b351-8a84-4ff2-8724-d86eae9b842e': 13, '4188d3a4-077d-46b7-9c86-23e1a036f6c1': 14, '455d3c66-7dc6-4537-a39a-36d3e9119df7': 15, '46407397-a7d5-4c6b-92c6-dbe038b1457b': 16, '4bcb1253-a636-4df4-8cb0-a35c04dfef31': 17, '51b11269-2ca8-4b2a-9163-f21758420e78': 18, '550ce7e7-747b-495f-b122-acdc4d0b8e54': 19, '5d901039-a89c-4bfb-967b-bf66f4df075e': 20, '6054afcb-5bab-4702-90a0-b259b5d3217c': 21, '6d72aad6-187a-4392-a4c4-ed87269c51cf': 22, '6f4073b8-d8ea-4ade-8a18-c5d1d5d5aa9a': 23, '6f81754e-285d-4ce0-b59e-af7edb02d108': 24, '74d5859f-ed66-4d3e-aa0e-93d7a592ce41': 25, '7a4e4bc8-922c-4c84-865c-25ba34136be1': 26, '7efeb4b1-3d19-4762-b163-63328d66303b': 27, '897e3b53-5d4d-444b-85cb-2cdc8a97d903': 28, '8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14': 29, '8e116af7-7db7-4e35-a68b-b0939c066c78': 30, '9ec204e4-f0a3-42f8-8458-b772a6797cab': 31, 'a097acff-6266-4291-9fbd-137af7ecd439': 32, 'a82b78bb-7fde-4cb3-94a4-035baf10bcf0': 33, 'a9f325aa-8c05-4e4f-8341-9e4358565f4f': 34, 'abed40dc-063f-4598-8ba5-9fe749c0615d': 35, 'b21acd93-60fd-4127-8a43-2f5178f4a830': 36, 'b5062e3e-641c-4e3a-907b-ac864d2e7652': 37, 'c867c42d-a52d-4a24-8ae3-f75d256b5618': 38, 'ce88f674-ab7a-43da-9201-468d38539e4a': 39, 'd1acdb87-bb67-4f30-84aa-990e56a09c92': 40, 'da52d699-e8d2-4dc5-9191-a2199e0b6a9b': 41, 'deec51c9-3b1e-4b9e-993c-4776f20e8bb2': 42, 'e2392362-125e-4f76-a2ee-524b183a3412': 43, 'e528b65e-1107-4b8c-8988-490e4fece599': 44, 'eb03d19a-b88d-4de4-8a64-ca0ac66f426b': 45, 'eb303e01-261e-4972-8c07-c9b4e7a4922a': 46, 'ecb0df7a-4e8d-4a03-b162-053391d3afaf': 47, 'ecc2413d-8a48-416e-a3a2-d30106ca36cb': 48, 'ecc2413d-8a48-416e-a3a2-d30106ca36cb-': 49, 'f918266a-b3e0-4914-865d-4faa564f1aef': 50}

count_dict = {}
num = 0

for filedir in os.listdir(filepath):
    print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if filedir[-2] == '-':
        task_id = filedir[:-4]
    else:
        task_id = filedir

    idx = dict[task_id]

    if idx in count_dict:
        count_dict[idx] += 1
    else:
        count_dict[idx] = 1
    # print("idx = ", idx)

    if filedir[-2] == '-':
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + filedir[-4:] + ".csv"
    else:
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + ".csv"

    print("csv_file = ", csv_file)
    if not os.path.exists(csv_file):
        num += 1
        continue

    # if len(open(csv_file).readlines()) <= 1:
    #     print("\nnow = ", filedir)
    #     num += 1
    #     os.remove(csv_file)
    #     continue

    try:
        data = pd.read_csv(csv_file)
    except:
        continue
    task_ID = []
    # sub_ID = []
    for index, row in data.iterrows():
        task_ID.append("OSWorld_Windows" + str(idx) + '_' + str(count_dict[idx]))
        # sub_ID.append(count_dict[idx])
    data['task_ID'] = task_ID
    # data['sub_ID'] = sub_ID
    data = data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
    # print("data = \n", data)
    final_data = pd.concat([final_data, data])
    # print("final_data = \n", final_data)

# print("\nnum = ", num)
final_data = final_data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
final_data.to_csv(final_csv_file)

