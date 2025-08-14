import os
import csv
import pandas as pd

csv_file = "Reward_Model_WebArena.csv"
final_csv_file = "Reward_Model_WebArena_modify.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

data = pd.read_csv(csv_file, encoding='iso-8859-1')
final_data = pd.read_csv(final_csv_file)

task_ID = []
# sub_ID = []
for index, row in data.iterrows():
    task_ID.append('WebArena_' + row['task_ID'])
    # sub_ID.append(count_dict[idx])
data['task_ID'] = task_ID
final_data = pd.concat([final_data, data])

final_data = final_data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
final_data.to_csv(final_csv_file)