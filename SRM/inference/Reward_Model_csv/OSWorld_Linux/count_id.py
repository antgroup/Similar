import os
import csv
import pandas as pd

csv_file = "Reward_Model_OSWorld_Linux.csv"

final_data = pd.read_csv(csv_file, encoding="gbk")

task_id_list = []
subtask_id_list = []
subtask_id_useful_list = []
task_id_useful_list = []

for index, row in final_data.iterrows():
    subtask_id = row['task_ID']
    if subtask_id not in task_id_list:
        subtask_id_list.append(subtask_id)

    if row['IP'] != 0 and subtask_id not in subtask_id_useful_list:
        subtask_id_useful_list.append(subtask_id)
        task_id = int(subtask_id[len("OSWorld_Linux_") : ].split('_')[0])
        if task_id not in task_id_useful_list:
            task_id_useful_list.append(task_id)

    task_id = int(subtask_id[len("OSWorld_Linux_") : ].split('_')[0])

    if task_id not in task_id_list:
        task_id_list.append(task_id)

task_id_list = list(set(task_id_list))
subtask_id_list = list(set(subtask_id_list))
task_id_useful_list.sort()
# task_id_useful_list = list(set(task_id_useful_list))
print("len(task_id_list) = ", len(task_id_list)) # 333
print("task_id_list = ", task_id_list)
print("\n")
print("len(subtask_id_useful_list) = ", len(subtask_id_useful_list)) # 74
# print("subtask_id_useful_list = ", subtask_id_useful_list)
print("\n")
print("len(task_id_useful_list) = ", len(task_id_useful_list)) # 25
print("task_id_useful_list = ", task_id_useful_list)

task_id_useful_list =  [2, 6, 8, 26, 41, 62, 74, 81, 83, 91, 109, 131, 158, 175, 186, 189, 209, 218, 234, 264, 296, 326, 327, 328, 337]