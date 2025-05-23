import os
import csv
import pandas as pd
import json

task_id_useful_list =  [2, 6, 8, 26, 41, 62, 74, 81, 83, 91, 109, 131, 158, 175, 186, 189, 209, 218, 234, 264, 296, 326, 327, 328, 337]
csv_file = "Reward_Model_OSWorld_Linux.csv"
json_file = "Reward_Model_OSWorld_Linux_useful.json"

data = pd.read_csv(csv_file, encoding='iso-8859-1')

dict = {}
if not os.path.exists(json_file):
    for (index, row) in data.iterrows():
        # idx = int(row['task_ID'][:row['task_ID'].find('_')])
        subtask_id = row['task_ID']
        idx = int(subtask_id[len("OSWorld_Linux_") : ].split('_')[0])
        if not idx in task_id_useful_list:
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

for id in task_id_useful_list:
    list_now = list(dict[str(id)].keys()) # 当前任务有多少条轨迹
    # print("length = ", len(list_now))

    traj_score_dict = {}

    for i in range(len(list_now)):
        step_idx = 0
        sum_traj_score = 0
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
            except:
                break
        # print("step_idx = ", step_idx)
        if step_idx:
            avg_traj_score = sum_traj_score / step_idx
            traj_score_dict[i] = avg_traj_score

    for i in range(len(list_now)):
        for j in range(i + 1, len(list_now)):
            now = str(list_now[i]) + "_vs_" + str(list_now[j])
            now_reverse = str(list_now[j]) + "_vs_" + str(list_now[i])
            # print(list_now[i], list_now[j]) # 当前在比较的轨迹
            min_len = min(len(dict[str(id)][list_now[i]]), len(dict[str(id)][list_now[j]]))
            # print("min_len = ", min_len)
            step_idx = 0
            for k in range(min_len):
                step_idx += 1
                try:
                    if dict[str(id)][list_now[i]][str(k + 1)]["action"] == dict[str(id)][list_now[j]][str(k + 1)]["action"]:
                        continue
                    else:
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

                        if answer1_IP > answer2_IP:
                            IP_list.append(now)
                        elif answer1_IP < answer2_IP:
                            IP_list.append(now_reverse)

                        if answer1_E > answer2_E:
                            E_list.append(now)
                        elif answer1_E < answer2_E:
                            E_list.append(now_reverse)

                        if answer1_TC > answer2_TC:
                            TC_list.append(now)
                        elif answer1_TC < answer2_TC:
                            TC_list.append(now_reverse)

                        if answer1_TR > answer2_TR:
                            TR_list.append(now)
                        elif answer1_TR > answer2_TR:
                            TR_list.append(now_reverse)

                        if answer1_C > answer2_C:
                            C_list.append(now)
                        elif answer1_C > answer2_C:
                            C_list.append(now_reverse)

                        if answer1_total > answer2_total:
                            all_list.append(now)
                        elif answer1_total > answer2_total:
                            all_list.append(now_reverse)

                        break

                except Exception as e:
                    break

            try:
                if traj_score_dict[i] > traj_score_dict[j]:
                    traj_list.append(now)
                elif traj_score_dict[i] < traj_score_dict[j]:
                    traj_list.append(now_reverse)
            except:
                continue


print("len(IP_list) = ", len(IP_list)) # 47
# print("IP_list = ", IP_list)
print("\n")
print("len(E_list) = ", len(E_list)) # 78
# print("E_list = ", E_list)
print("\n")
print("len(TC_list) = ", len(TC_list)) # 74
# print("TC_list = ", TC_list)
print("\n")
print("len(TR_list) = ", len(TR_list)) # 29
# print("TR_list = ", TR_list)
print("\n")
print("len(C_list) = ", len(C_list)) # 33
# print("E_list = ", E_list)
print("\n")
print("len(all_list) = ", len(all_list)) # 49
# print("all_list = ", all_list)
print("\n")
print("len(traj_list) = ", len(traj_list)) # 126
# print("traj_list = ", traj_list)