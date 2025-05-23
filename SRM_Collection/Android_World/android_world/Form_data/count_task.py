import os
import json

task_file = "android_world/task_metadata.json"

with open(task_file, 'r') as f:
    tasks = json.load(f)

num = 0
for task in tasks:
    num += 1
    if "VlcCreateTwoPlaylists" == task['task_name']:
        print("now = ", num)

print("num = ", num)
