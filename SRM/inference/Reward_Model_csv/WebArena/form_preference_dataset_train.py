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


csv_file = "Reward_Model_WebArena.csv"
json_file = "Reward_Model_WebArena_useful.json"
dataset_csv_file = "Reward_Model_WebArena_preference_dataset_train.csv"

task_id_useful_list =  [3, 22, 23, 24, 28, 29, 30, 41, 42, 43, 44, 69, 79, 94, 95, 115, 118, 128, 129, 130, 132, 134, 135, 149, 150, 156, 158, 160, 161, 164, 166, 167, 169, 177, 183, 188, 190, 198, 199, 202, 203, 205, 208, 209, 211, 212, 225, 227, 230, 231, 247, 258, 260, 261, 262, 264, 273, 274, 275, 276, 278, 302, 303, 305, 306, 308, 310, 311, 312, 313, 314, 315, 317, 318, 322, 326, 340, 348, 351, 354, 355, 358, 359, 360, 361, 368, 376, 384, 387, 388, 392, 395, 397, 447, 465, 468, 472, 474, 477, 478, 479, 491, 511, 512, 514, 515, 516, 517, 518, 533, 535, 539, 580, 581, 650, 651, 652, 691, 692, 693, 723, 726, 731, 753, 754, 772, 775, 784, 785, 787, 790, 793, 794, 795]
task_id_useful_test_list =  [491, 258, 794, 652, 274, 130, 731, 211, 479, 161, 190, 322, 693, 41, 132, 29, 384, 183, 517, 150, 149, 69, 355, 95, 387, 305]
task_id_useful_train_list = list(set(task_id_useful_list) - set(task_id_useful_test_list))


data = pd.read_csv(csv_file, encoding="iso-8859-1")
dict = {}
with open(json_file, 'r') as f:
    dict = json.load(f)

IP_list = []
E_list = []
TC_list = []
TR_list = []
C_list = []
all_list = []

with open(dataset_csv_file, 'w', newline='') as file:
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'chosen_score', 'rejected_score']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for id in task_id_useful_train_list:
        list_now = list(dict[str(id)].keys())
        # print("length = ", len(list_now))

        traj_dict = {}
        traj_score_dict = {}

        for i in range(len(list_now)):
            for j in range(i + 1, len(list_now)):
                now = "WebArena_" + str(list_now[i]) + "_vs_" + str(list_now[j])
                now_reverse = "WebArena_" + str(list_now[j]) + "_vs_" + str(list_now[i])
                # print(list_now[i], list_now[j])
                min_len = min(len(dict[str(id)][list_now[i]]), len(dict[str(id)][list_now[j]]))
                # print("min_len = ", min_len)

                reason_steps = ''
                step_idx = 0

                for k in range(min_len):
                    step_idx += 1
                    try:
                        if dict[str(id)][list_now[i]][str(k + 1)]["action"] == dict[str(id)][list_now[j]][str(k + 1)]["action"]:
                            reason_steps += "Step " + str(step_idx) + " : " + dict[str(id)][list_now[i]][str(k + 1)]["action"] + "\n\n"
                            continue
                        else:
                            instruction = dict[str(id)][list_now[i]][str(k + 1)]["instruction"]
                            observation_url = dict[str(id)][list_now[i]][str(k + 1)]["observation_url"]

                            answer1_IP = dict[str(id)][list_now[i]][str(k + 1)]["IP"]
                            answer1_E = dict[str(id)][list_now[i]][str(k + 1)]["E"]
                            answer1_TC = dict[str(id)][list_now[i]][str(k + 1)]["TC"]
                            answer1_TR = dict[str(id)][list_now[i]][str(k + 1)]["TR"]
                            answer1_C = dict[str(id)][list_now[i]][str(k + 1)]["C"]
                            answer1_total = answer1_IP * 5 + answer1_E * 5 + answer1_TC * 3 + answer1_TR + answer1_C
                            action1 = dict[str(id)][list_now[i]][str(k + 1)]["action"]

                            answer2_IP = dict[str(id)][list_now[j]][str(k + 1)]["IP"]
                            answer2_E = dict[str(id)][list_now[j]][str(k + 1)]["E"]
                            answer2_TC = dict[str(id)][list_now[j]][str(k + 1)]["TC"]
                            answer2_TR = dict[str(id)][list_now[j]][str(k + 1)]["TR"]
                            answer2_C = dict[str(id)][list_now[j]][str(k + 1)]["C"]
                            answer2_total = answer2_IP * 5 + answer2_E * 5 + answer2_TC * 3 + answer2_TR + answer2_C
                            action2 = dict[str(id)][list_now[j]][str(k + 1)]["action"]

                            user_message = user_prompt
                            user_message = user_message.replace("<INSTRUCTION>", instruction)
                            user_message = user_message.replace("<REASON_STEPS>", reason_steps)

                            messages1 = [
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
                                    "content": action1,
                                }
                            ]

                            messages2 = [
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
                                    "content": action2,
                                }
                            ]

                            if answer1_IP > answer2_IP:
                                IP_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_IP
                                rejected_score = answer2_IP
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'IP',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_IP < answer2_IP:
                                IP_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_IP
                                rejected_score = answer1_IP
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'IP',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            if answer1_E > answer2_E:
                                E_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_E
                                rejected_score = answer2_E
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'E',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_E < answer2_E:
                                E_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_E
                                rejected_score = answer1_E
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'E',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            if answer1_TC > answer2_TC:
                                TC_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_TC
                                rejected_score = answer2_TC
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TC',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_TC < answer2_TC:
                                TC_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_TC
                                rejected_score = answer1_TC
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TC',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            if answer1_TR > answer2_TR:
                                TR_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_TR
                                rejected_score = answer2_TR
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TR',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_TR > answer2_TR:
                                TR_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_TR
                                rejected_score = answer1_TR
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TR',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            if answer1_C > answer2_C:
                                C_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_C
                                rejected_score = answer2_C
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'C',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_C > answer2_C:
                                C_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_C
                                rejected_score = answer1_C
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'C',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})


                            if answer1_total > answer2_total:
                                all_list.append(now)
                                chosen = messages1
                                rejected = messages2
                                chosen_score = answer1_total
                                rejected_score = answer2_total
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'total',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})
                            elif answer1_total > answer2_total:
                                all_list.append(now_reverse)
                                chosen = messages2
                                rejected = messages1
                                chosen_score = answer2_total
                                rejected_score = answer1_total
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'total',
                                     'reason_steps': reason_steps, 'image_url': observation_url,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            break

                    except Exception as e:
                        break

print("len(IP_list) = ", len(IP_list)) # 514
# print("IP_list = ", IP_list)
print("\n")
print("len(E_list) = ", len(E_list)) # 1232
# print("E_list = ", E_list)
print("\n")
print("len(TC_list) = ", len(TC_list)) # 1225
# print("TC_list = ", TC_list)
print("\n")
print("len(TR_list) = ", len(TR_list)) # 237
# print("TR_list = ", TR_list)
print("\n")
print("len(C_list) = ", len(C_list)) # 174
# print("E_list = ", E_list)
print("\n")
print("len(all_list) = ", len(all_list)) # 609
# print("all_list = ", all_list)