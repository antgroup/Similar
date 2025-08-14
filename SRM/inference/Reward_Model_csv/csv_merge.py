import os
import csv
import pandas as pd
import json

# list = ["Android_World/Reward_Model_Android_World_test_12.10.csv", "VisualWebArena/Reward_Model_VisualWebArena_test_12.10.csv", "WebArena/Reward_Model_WebArena_test_12.10.csv"]
list = ["Android_World/Reward_Model_Android_World_test_2_12.22.csv", "VisualWebArena/Reward_Model_VisualWebArena_test_2_12.22.csv", "WebArena/Reward_Model_WebArena_test_2_12.22.csv", "OSWorld_Linux/Reward_Model_OSWorld_Linux_test_2_12.22.csv", "OSWorld_Windows/Reward_Model_OSWorld_Windows_test_2_12.22.csv"]


# final_csv_file = "Reward_Model_test_12.10.csv"
final_csv_file = "Reward_Model_test_2_12.22.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)

for filepath in list:
    csv_file = filepath
    temp = pd.read_csv(csv_file)
    final_data = pd.concat([final_data, temp])

final_data = final_data[['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected']]
final_data.to_csv(final_csv_file)