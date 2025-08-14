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


csv_file = "Reward_Model_VisualWebArena_original.csv"
json_file = "Reward_Model_VisualWebArena_useful.json"
dataset_csv_file = "Reward_Model_VisualWebArena_preference_dataset_test_modify.csv"

task_id_useful_list =  ['classifieds_10', 'classifieds_106', 'classifieds_11', 'classifieds_111', 'classifieds_118', 'classifieds_120', 'classifieds_121', 'classifieds_124', 'classifieds_125', 'classifieds_127', 'classifieds_129', 'classifieds_130', 'classifieds_135', 'classifieds_140', 'classifieds_15', 'classifieds_153', 'classifieds_164', 'classifieds_167', 'classifieds_173', 'classifieds_174', 'classifieds_189', 'classifieds_194', 'classifieds_195', 'classifieds_196', 'classifieds_199', 'classifieds_209', 'classifieds_220', 'classifieds_24', 'classifieds_29', 'classifieds_49', 'classifieds_50', 'classifieds_53', 'classifieds_98', 'reddit_120', 'reddit_155', 'reddit_160', 'reddit_162', 'reddit_173', 'reddit_182', 'reddit_19', 'reddit_196', 'reddit_31', 'reddit_36', 'reddit_39', 'reddit_69', 'reddit_7', 'reddit_92', 'shopping_111', 'shopping_128', 'shopping_129', 'shopping_14', 'shopping_147', 'shopping_148', 'shopping_161', 'shopping_167', 'shopping_188', 'shopping_200', 'shopping_204', 'shopping_209', 'shopping_212', 'shopping_217', 'shopping_58', 'shopping_69', 'shopping_70', 'shopping_71', 'shopping_73', 'shopping_80', 'shopping_81', 'shopping_91']
task_id_useful_test_list =  ['shopping_129', 'classifieds_125', 'classifieds_173', 'shopping_204', 'classifieds_49', 'classifieds_127', 'classifieds_53', 'shopping_111', 'reddit_182', 'shopping_217', 'shopping_212', 'reddit_162', 'classifieds_153']
# task_id_useful_train_list = list(set(task_id_useful_list) - set(task_id_useful_test_list))


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
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen_action', 'rejected_action', 'chosen', 'rejected', 'chosen_score', 'rejected_score']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for id in task_id_useful_test_list:
        list_now = list(dict[str(id)].keys())
        # print("length = ", len(list_now))

        traj_dict = {}
        traj_score_dict = {}

        for i in range(len(list_now)):
            for j in range(i + 1, len(list_now)):
                now = "VisualWebArena_" + str(list_now[i]) + "_vs_" + str(list_now[j])
                now_reverse = "VisualWebArena_" + str(list_now[j]) + "_vs_" + str(list_now[i])
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

                            answer1_IP = dict[str(id)][list_now[i]][str(k + 1)]["IP"] * 5
                            answer1_E = dict[str(id)][list_now[i]][str(k + 1)]["E"] * 5
                            answer1_TC = dict[str(id)][list_now[i]][str(k + 1)]["TC"] * 3
                            answer1_TR = dict[str(id)][list_now[i]][str(k + 1)]["TR"]
                            answer1_C = dict[str(id)][list_now[i]][str(k + 1)]["C"]
                            answer1_total = answer1_IP + answer1_E + answer1_TC + answer1_TR + answer1_C
                            action1 = dict[str(id)][list_now[i]][str(k + 1)]["action"]

                            answer2_IP = dict[str(id)][list_now[j]][str(k + 1)]["IP"] * 5
                            answer2_E = dict[str(id)][list_now[j]][str(k + 1)]["E"] * 5
                            answer2_TC = dict[str(id)][list_now[j]][str(k + 1)]["TC"] * 3
                            answer2_TR = dict[str(id)][list_now[j]][str(k + 1)]["TR"]
                            answer2_C = dict[str(id)][list_now[j]][str(k + 1)]["C"]
                            answer2_total = answer2_IP + answer2_E + answer2_TC + answer2_TR + answer2_C
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
                                     'chosen_action': action1, 'rejected_action': action2,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action1, 'rejected_action': action2,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
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
                                     'chosen_action': action2, 'rejected_action': action1,
                                     'chosen': chosen, 'rejected': rejected,
                                     'chosen_score': chosen_score, 'rejected_score': rejected_score})

                            break

                    except Exception as e:
                        break

print("len(IP_list) = ", len(IP_list)) # 379
# print("IP_list = ", IP_list)
print("\n")
print("len(E_list) = ", len(E_list)) # 4666
# print("E_list = ", E_list)
print("\n")
print("len(TC_list) = ", len(TC_list)) # 4658
# print("TC_list = ", TC_list)
print("\n")
print("len(TR_list) = ", len(TR_list)) # 326
# print("TR_list = ", TR_list)
print("\n")
print("len(C_list) = ", len(C_list)) # 1572
# print("E_list = ", E_list)
print("\n")
print("len(all_list) = ", len(all_list)) # 2730
# print("all_list = ", all_list)