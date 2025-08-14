import os
import json

task_file = "evaluation_examples/test_all.json"
with open(task_file) as f:
    tasks = json.load(f)

num_all = 0
for (key, value) in tasks.items():
    print("domain now = ", key)
    num = 0
    list_now = value
    for task_id in list_now:
        num += 1
    print("num = ", num)
    num_all += num
print("num_all = ", num_all)
