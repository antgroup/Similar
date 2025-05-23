import os
import csv
import pandas as pd
from random import sample

csv_file = "Reward_Model_original.csv"

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
        task_id = int(subtask_id[len("Android_World_") : ].split('_')[0])
        if task_id not in task_id_useful_list:
            task_id_useful_list.append(task_id)

    task_id = int(subtask_id[len("Android_World_") : ].split('_')[0])

    if task_id not in task_id_list:
        task_id_list.append(task_id)

task_id_list = list(set(task_id_list))
subtask_id_list = list(set(subtask_id_list))
task_id_useful_list.sort()
# task_id_useful_list = list(set(task_id_useful_list))
# task_id_useful_test_list = sample(task_id_useful_list, int(len(task_id_useful_list) * 0.2))
task_id_useful_test_list =  [58, 26, 11, 61, 71, 7, 21, 19, 87, 12]
task_id_rest_list = list(set(task_id_useful_list) - set(task_id_useful_test_list))
task_id_useful_test_2_list = sample(task_id_rest_list, int(len(task_id_rest_list) * 0.2))

print("len(task_id_list) = ", len(task_id_list)) # 108
print("task_id_list = ", task_id_list)
print("\n")
print("len(subtask_id_useful_list) = ", len(subtask_id_useful_list)) # 1004
# print("subtask_id_useful_list = ", subtask_id_useful_list)
print("\n")
print("len(task_id_useful_list) = ", len(task_id_useful_list)) # 52
print("task_id_useful_list = ", task_id_useful_list)
print("\n")
print("len(task_id_useful_test_list) = ", len(task_id_useful_test_list)) # 10
print("task_id_useful_test_list = ", task_id_useful_test_list)
print("\n")
print("len(task_id_useful_test_2_list) = ", len(task_id_useful_test_2_list)) # 8
print("task_id_useful_test_2_list = ", task_id_useful_test_2_list)

task_id_useful_list =  [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 26, 28, 29, 30, 31, 33, 36, 37, 38, 42, 45, 47, 49, 50, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 71, 72, 76, 81, 87, 99, 104, 109]

task_id_useful_test_list =  [58, 26, 11, 61, 71, 7, 21, 19, 87, 12]

task_id_useful_test_2_list =  [28, 3, 47, 55, 20, 104, 31, 50]