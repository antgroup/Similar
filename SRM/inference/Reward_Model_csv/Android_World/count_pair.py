import os
import csv
import pandas as pd
import json

task_id_useful_list =  [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 26, 28, 29, 30, 31, 33, 36, 37, 38, 42, 45, 47, 49, 50, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 64, 65, 66, 71, 72, 76, 81, 87, 99, 104, 109]
csv_file = "Reward_Model_original.csv"
json_file = "Reward_Model_Android_World_useful.json"

data = pd.read_csv(csv_file, encoding='iso-8859-1')

dict = {}
if not os.path.exists(json_file):
    for (index, row) in data.iterrows():
        # idx = int(row['task_ID'][:row['task_ID'].find('_')])
        subtask_id = row['task_ID']
        idx = int(subtask_id[len("Android_World_") : ].split('_')[0])

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


print("len(IP_list) = ", len(IP_list)) # 4903   4903
# print("IP_list = ", IP_list)
print("\n")
print("len(E_list) = ", len(E_list)) # 7099   10566
# print("E_list = ", E_list)
print("\n")
print("len(TC_list) = ", len(TC_list)) # 5732   10553
# print("TC_list = ", TC_list)
print("\n")
print("len(TR_list) = ", len(TR_list)) # 807   807
# print("TR_list = ", TR_list)
print("\n")
print("len(C_list) = ", len(C_list)) # 1003   1003
# print("E_list = ", E_list)
print("\n")
print("len(all_list) = ", len(all_list)) # 4302   5095
# print("all_list = ", all_list)
print("\n")
print("len(traj_list) = ", len(traj_list)) # 16600
# print("traj_list = ", traj_list)