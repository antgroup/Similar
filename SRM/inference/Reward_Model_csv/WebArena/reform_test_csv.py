import os
import pandas as pd
import csv
import json

import requests
from PIL import Image
from io import BytesIO

answer_template = '''
STEP <STEP_IDX> : <ACTION> \n
'''

csv_file = "Reward_Model_WebArena.csv"
json_file = "Reward_Model_WebArena_useful.json"
# reform_csv_file = "Reward_Model_WebArena_test_12.10.csv"
reform_csv_file = "Reward_Model_WebArena_test_2_12.22.csv"

data = pd.read_csv(csv_file, encoding='iso-8859-1')

# task_id_useful_test_list =  [491, 258, 794, 652, 274, 130, 731, 211, 479, 161, 190, 322, 693, 41, 132, 29, 384, 183, 517, 150, 149, 69, 355, 95, 387, 305]
task_id_useful_test_2_list =  [351, 318, 580, 167, 314, 135, 306, 3, 692, 158, 225, 169, 397, 310, 368, 535, 723, 278, 28, 539, 478]
# completed_id =  [115, 118, 126, 128, 132, 134, 135, 149, 150, 156, 158, 160, 161, 164, 166, 167, 169, 177, 183, 188, 190, 198, 199, 202, 203, 205, 208, 209, 211, 212, 22, 225, 227, 23, 230, 231, 233, 239, 24, 247, 25, 258, 260, 261, 262, 264, 269, 271, 273, 274, 275, 276, 278, 28, 29, 298, 3, 30, 302, 303, 305, 306, 308, 310, 311, 312, 313, 314, 315, 317, 318, 322, 326, 340, 344, 348, 350, 351, 354, 355, 358, 359, 360, 361, 362, 368, 376, 384, 387, 388, 392, 395, 397, 41, 42, 43, 44, 447]


dict = {}

if not os.path.exists(json_file):
    for (index, row) in data.iterrows():
        idx = int(row['task_ID'][:row['task_ID'].find('_')])
        if not idx in task_id_useful_test_2_list:
            continue
        if str(idx) not in dict:
            dict[str(idx)] = {}
        if str(row['task_ID']) not in dict[str(idx)]:
            dict[str(idx)][str(row['task_ID'])] = {}
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])] = {}
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['instruction'] = row['instruction']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['observation_url'] = row['observation_url']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['action'] = row['action']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['IP'] = row['IP']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['E'] = row['E']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['TC'] = row['TC']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['TR'] = row['TR']
        dict[str(idx)][str(row['task_ID'])][str(row['step_idx'])]['C'] = row['C']

    with open(json_file, "w") as file:
        json.dump(dict, file)
else:
    with open(json_file, 'r') as f:
        dict = json.load(f)


IP_list = []
E_list = []
TC_list = []
TR_list = []
C_list = []
all_list = []
traj_list = []

with open(reform_csv_file, 'w', newline='') as file:
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for id in task_id_useful_test_2_list:
        list_now = list(dict[str(id)].keys())
        # print("length = ", len(list_now))

        traj_dict = {}
        traj_score_dict = {}

        for i in range(len(list_now)):
            step_idx = 0
            sum_traj_score = 0
            reason_steps = ''
            for k in range(len(dict[str(id)][list_now[i]])):
                try:
                    if dict[str(id)][list_now[i]][str(k + 1)]["action"]:
                        flag = 1

                    answer1_IP = dict[str(id)][list_now[i]][str(k + 1)]["IP"]
                    answer1_E = dict[str(id)][list_now[i]][str(k + 1)]["E"]
                    answer1_TC = dict[str(id)][list_now[i]][str(k + 1)]["TC"]
                    answer1_TR = dict[str(id)][list_now[i]][str(k + 1)]["TR"]
                    answer1_C = dict[str(id)][list_now[i]][str(k + 1)]["C"]
                    sum_traj_score += answer1_IP + answer1_E + answer1_TC + answer1_TR + answer1_C

                    step_idx += 1
                    reason_steps += "Step " + str(step_idx) + " : " + dict[str(id)][list_now[i]][str(k + 1)]["action"] + "\n\n"
                except:
                    break
            # print("step_idx = ", step_idx)
            if step_idx:
                avg_traj_score = sum_traj_score / step_idx
                traj_score_dict[i] = avg_traj_score
                traj_dict[i] = reason_steps


        for i in range(len(list_now)):
            for j in range(i + 1, len(list_now)):
                now = "WebArena_" + str(list_now[i]) + "_vs_" + str(list_now[j])
                now_reverse = "WebArena_" + str(list_now[j]) + "_vs_" + str(list_now[i])
                # print(list_now[i], list_now[j])
                min_len = min(len(dict[str(id)][list_now[i]]), len(dict[str(id)][list_now[j]]))
                # print("min_len = ", min_len)
                
                reason_steps = 'Let\'s complete the task step by step. \n\n'
                step_idx = 0

                for k in range(min_len):
                    step_idx += 1
                    try:
                        if dict[str(id)][list_now[i]][str(k + 1)]["action"] == dict[str(id)][list_now[j]][str(k + 1)]["action"]:
                            reason_steps += "Step " + str(step_idx) + " : " + dict[str(id)][list_now[i]][str(k + 1)]["action"] + "\n\n"
                            continue
                        else:
                            instruction = dict[str(id)][list_now[i]][str(k + 1)]["instruction"]
                            image_url = dict[str(id)][list_now[i]][str(k + 1)]["observation_url"]

                            answer1_IP = dict[str(id)][list_now[i]][str(k + 1)]["IP"]
                            answer1_E = dict[str(id)][list_now[i]][str(k + 1)]["E"]
                            answer1_TC = dict[str(id)][list_now[i]][str(k + 1)]["TC"]
                            answer1_TR = dict[str(id)][list_now[i]][str(k + 1)]["TR"]
                            answer1_C = dict[str(id)][list_now[i]][str(k + 1)]["C"]
                            answer1_total = answer1_IP * 5 + answer1_E * 5 + answer1_TC * 3 + answer1_TR + answer1_C

                            answer2_IP = dict[str(id)][list_now[j]][str(k + 1)]["IP"]
                            answer2_E = dict[str(id)][list_now[j]][str(k + 1)]["E"]
                            answer2_TC = dict[str(id)][list_now[j]][str(k + 1)]["TC"]
                            answer2_TR = dict[str(id)][list_now[j]][str(k + 1)]["TR"]
                            answer2_C = dict[str(id)][list_now[j]][str(k + 1)]["C"]
                            answer2_total = answer2_IP * 5 + answer2_E * 5 + answer2_TC * 3 + answer2_TR + answer2_C

                            answer1 = answer_template
                            answer1 = answer1.replace("<STEP_IDX>", str(step_idx))
                            answer1 = answer1.replace("<ACTION>", str(dict[str(id)][list_now[i]][str(k + 1)]["action"]))

                            answer2 = answer_template
                            answer2 = answer2.replace("<STEP_IDX>", str(step_idx))
                            answer2 = answer2.replace("<ACTION>", dict[str(id)][list_now[j]][str(k + 1)]["action"])

                            if answer1_IP > answer2_IP:
                                IP_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'IP',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_IP < answer2_IP:
                                IP_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'IP',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})

                            if answer1_E > answer2_E:
                                E_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'E',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_E < answer2_E:
                                E_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'E',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})

                            if answer1_TC > answer2_TC:
                                TC_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TC',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_TC < answer2_TC:
                                TC_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TR',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})

                            if answer1_TR > answer2_TR:
                                TR_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TR',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_TR > answer2_TR:
                                TR_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'TR',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})

                            if answer1_C > answer2_C:
                                C_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'C',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_C > answer2_C:
                                C_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'C',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})


                            if answer1_total > answer2_total:
                                all_list.append(now)
                                chosen = answer1
                                rejected = answer2
                                writer.writerow(
                                    {'compare_id': now, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'total',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})
                            elif answer1_total > answer2_total:
                                all_list.append(now_reverse)
                                chosen = answer2
                                rejected = answer1
                                writer.writerow(
                                    {'compare_id': now_reverse, 'step_idx': step_idx,
                                     'instruction': instruction, 'type': 'total',
                                     'reason_steps': reason_steps, 'image_url': image_url,
                                     'chosen': chosen, 'rejected': rejected})

                            break

                    except Exception as e:
                        break

                try:
                    instruction = dict[str(id)][list_now[i]]["1"]["instruction"]
                    image_url = dict[str(id)][list_now[i]]["1"]["observation_url"]
                    if traj_score_dict[i] > traj_score_dict[j]:
                        traj_list.append(now)
                        writer.writerow(
                            {'compare_id': now, 'step_idx': -1,
                             'instruction': instruction, 'type': 'traj',
                             'reason_steps': 'Let\'s complete the task step by step. \n', 'image_url': image_url,
                             'chosen': traj_dict[i], 'rejected': traj_dict[j]})
                    elif traj_score_dict[i] < traj_score_dict[j]:
                        traj_list.append(now_reverse)
                        writer.writerow(
                            {'compare_id': now, 'step_idx': -1,
                             'instruction': instruction, 'type': 'traj',
                             'reason_steps': 'Let\'s complete the task step by step. \n', 'image_url': image_url,
                             'chosen': traj_dict[j], 'rejected': traj_dict[i]})
                except:
                    continue

print("len(IP_list) = ", len(IP_list)) # 169  119
# print("IP_list = ", IP_list)
print("\n")
print("len(E_list) = ", len(E_list)) # 247  196
# print("E_list = ", E_list)
print("\n")
print("len(TC_list) = ", len(TC_list)) # 239  193
# print("TC_list = ", TC_list)
print("\n")
print("len(TR_list) = ", len(TR_list)) # 41  35
# print("TR_list = ", TR_list)
print("\n")
print("len(C_list) = ", len(C_list)) # 50  44
# print("E_list = ", E_list)
print("\n")
print("len(all_list) = ", len(all_list)) # 110  90
# print("all_list = ", all_list)
print("\n")
print("len(traj_list) = ", len(traj_list)) # 369  330
# print("traj_list = ", traj_list)