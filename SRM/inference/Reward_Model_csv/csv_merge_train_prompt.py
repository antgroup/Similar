import os
import csv
import pandas as pd
import json

list = ["Android_World/Reward_Model_Android_World_preference_dataset_train_prompt.csv", "VisualWebArena/Reward_Model_VisualWebArena_preference_dataset_train_prompt.csv", "WebArena/Reward_Model_WebArena_preference_dataset_train_prompt.csv"]

final_csv_file = "Reward_Model_preference_dataset_train_prompt.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen_action', 'rejected_action', 'chosen', 'rejected', 'chosen_score', 'rejected_score']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)

for filepath in list:
    csv_file = filepath
    temp = pd.read_csv(csv_file)
    final_data = pd.concat([final_data, temp])

final_data = final_data[['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen_action', 'rejected_action', 'chosen', 'rejected', 'chosen_score', 'rejected_score']]
final_data.to_csv(final_csv_file)