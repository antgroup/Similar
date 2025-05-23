import os
import csv
import pandas as pd
import json

list = ["Android_World/Reward_Model_Android_World_dataset.csv", "VisualWebArena/Reward_Model_VisualWebArena_dataset.csv", "WebArena/Reward_Model_WebArena_dataset.csv"]

final_csv_file = "Reward_Model_multi-objective_dataset.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['task_id', 'instruction', 'step_idx', 'observation_url', 'action', 'messages', 'IP', 'E', 'TC', 'TR', 'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)

for filepath in list:
    csv_file = filepath
    temp = pd.read_csv(csv_file)
    final_data = pd.concat([final_data, temp])

final_data = final_data[['task_id', 'instruction', 'step_idx', 'observation_url', 'action', 'messages', 'IP', 'E', 'TC', 'TR', 'C']]
final_data.to_csv(final_csv_file)