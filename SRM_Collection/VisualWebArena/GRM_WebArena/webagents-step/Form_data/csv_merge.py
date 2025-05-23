import os
import csv
import pandas as pd

list = ["generate_data/tasks/vwa/test_shopping", "generate_data/tasks/vwa/test_classifieds", "generate_data/tasks/vwa/test_reddit"]

final_csv_file = "generate_data/Reward_Model_VisualWebArena_original.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, mode='w', newline='') as file:
        fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)

for filepath in list:
    csv_file = filepath + '/Reward_Model.csv'
    temp = pd.read_csv(csv_file)
    final_data = pd.concat([final_data, temp])

final_data = final_data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
final_data.to_csv(final_csv_file)