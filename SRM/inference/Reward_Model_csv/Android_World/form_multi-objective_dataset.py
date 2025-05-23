import os
import pandas as pd
import csv
import json


syestem_prompt = """
You are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks 
(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile 
devices, operation systems, etc.) based on given instructions.

You will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:
1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.
2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).
3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from 
which you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.
"""

user_prompt = """
[INSTRUCTION]: <INSTRUCTION>
[OBSERVATION]: which is a single image provided.
[REASON_STEPS]: <REASON_STEPS>
"""


csv_file = "Reward_Model_original.csv"
json_file = "Reward_Model_Android_World_useful.json"
dataset_csv_file = "Reward_Model_Android_World_dataset.csv"

task_id_useful_list =  [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 26, 28, 29, 30, 31, 33, 36, 37, 38, 42, 45, 47, 49, 50, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 71, 72, 76, 81, 87, 99, 104, 109]
task_id_useful_test_list =  [58, 26, 11, 61, 71, 7, 21, 19, 87, 12]
task_id_useful_train_list = list(set(task_id_useful_list) - set(task_id_useful_test_list))


data = pd.read_csv(csv_file, encoding="iso-8859-1")
dict = {}
with open(json_file, 'r') as f:
    dict = json.load(f)


with open(dataset_csv_file, 'w', newline='') as file:
    fields = ['task_id', 'instruction', 'step_idx', 'observation_url', 'action', 'messages', 'IP', 'E', 'TC', 'TR', 'C']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for id in task_id_useful_train_list:
        list_now = list(dict[str(id)].keys())
        # print("length = ", len(list_now))

        for i in range(len(list_now)):
            reason_steps = ''
            step_idx = 0
            for k in range(len(dict[str(id)][list_now[i]])):
                step_idx += 1
                try:
                    task_id = list_now[i] + "_" + str(k + 1)
                    instruction = dict[str(id)][list_now[i]][str(k + 1)]["instruction"]
                    observation_url = dict[str(id)][list_now[i]][str(k + 1)]["observation_url"]
                    action = dict[str(id)][list_now[i]][str(k + 1)]["action"]
                    IP = dict[str(id)][list_now[i]][str(k + 1)]["IP"] * 5
                    E = dict[str(id)][list_now[i]][str(k + 1)]["E"] * 5
                    TC = dict[str(id)][list_now[i]][str(k + 1)]["TC"] * 3
                    TR = dict[str(id)][list_now[i]][str(k + 1)]["TR"]
                    C = dict[str(id)][list_now[i]][str(k + 1)]["C"]

                    user_message = user_prompt
                    user_message = user_message.replace("<INSTRUCTION>", instruction)
                    user_message = user_message.replace("<REASON_STEPS>", reason_steps)

                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": syestem_prompt,
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_message,
                                },
                                {
                                    "type": "image",
                                    "image": observation_url,
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": action,
                        }
                    ]

                    reason_steps += "Step " + str(step_idx) + " : " + action + "\n\n"

                    writer.writerow(
                        {'task_id': task_id, 'instruction': instruction, 'step_idx': step_idx,
                         'observation_url': observation_url, 'action': action, 'messages': messages,
                         'IP': IP, 'E': E, 'TC': TC, 'TR': TR, 'C': C})
                except:
                    break