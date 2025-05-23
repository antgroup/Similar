import os
import csv
import pandas as pd

filepath = "generate_data/tasks/vwa/test_reddit" # 标注数据文件夹

final_csv_file = filepath + "/" + "Reward_Model.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        # fields = ['task_ID', 'sub_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR',
                  'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)
# print("final_data = \n", final_data)

# count_dict = {}
num = 0

for filedir in os.listdir(filepath):
    # print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if '-' in filedir:
        idx = int(filedir[:filedir.find('-')])
    else:
        idx = int(filedir)

    # if idx in count_dict:
    #     count_dict[idx] += 1
    # else:
    #     count_dict[idx] = 1
    # print("idx = ", idx)

    csv_file = os.path.join(filepath, filedir) + "/" + filedir + ".csv"
    if not os.path.exists(csv_file):
        num += 1
        continue

    # if len(open(csv_file).readlines()) <= 1:
    #     print("\nnow = ", filedir)
    #     num += 1
    #     os.remove(csv_file)
    #     continue

    # if (idx == 22):
    # print("\nnow = ", filedir)
    # print("count_dict[%d] = %d" % (idx, count_dict[idx]))

    data = pd.read_csv(csv_file)
    task_ID = []
    # sub_ID = []
    for index, row in data.iterrows():
        task_ID.append('VisualWebArena_reddit_' + filedir)
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

