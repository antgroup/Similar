import os
import pandas as pd
import csv
import json

csv_file = "Reward_Model_Android_World.csv"
csv_useful_file = "Reward_Model_Android_World_useful.csv"
csv_useful_train_file = "Reward_Model_Android_World_useful_train.csv"

task_id_useful_list =  [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 26, 28, 29, 30, 31, 33, 36, 37, 38, 42, 45, 47, 49, 50, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 71, 72, 76, 81, 87, 99, 104, 109]
task_id_useful_test_list =  [58, 26, 11, 61, 71, 7, 21, 19, 87, 12]
subtask_id_useful_list = []
subtask_id_useful_train_list = []

data = pd.read_csv(csv_file, encoding="iso-8859-1")

for index, row in data.iterrows():
    subtask_id = row['task_ID']
    task_id = int(subtask_id[len("Android_World_") : ].split('_')[0])

    if (task_id in task_id_useful_list):
        subtask_id_useful_list.append(subtask_id)
        if (task_id not in task_id_useful_test_list):
            subtask_id_useful_train_list.append(subtask_id)

csv_useful_data = data.loc[data['task_ID'].isin(subtask_id_useful_list)]
csv_useful_data.to_csv(csv_useful_file, index=False)

csv_useful_train_data = data.loc[data['task_ID'].isin(subtask_id_useful_train_list)]
csv_useful_train_data.to_csv(csv_useful_train_file, index=False)



