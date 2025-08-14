import os
import csv
import pandas as pd
import json
import chardet

def ask_chatgpt_ant(messages):
    param = get_default_config(model="gpt-4o")
    param["queryConditions"]["model"] = "gpt-4o"
    param["queryConditions"]["temperature"] = "0.2"

    param["queryConditions"]["messages"] = messages
    try:
        response = ask_chatgpt(param)
        # print(response)
        return response
    except Exception as e:
        print("error: ", e)
        return False


system_prompt = '''
    <Task Context>
    You're helping improve training data for a virtual agent that completes tasks across multiple platforms. Given a current state (screenshot + interaction history), we need to generate a rejected action that is clearly inferior to the chosen action for progressing toward the task goal. This rejected action must:
    + Be contextually plausible but ineffective when executed
    + NOT duplicate existing actions in the provided rejected action set
    + Clearly hinder task completion compared to the chosen action
    </Task Context>

    <Input Structure>
    + Instruction: [Task description]
    + Step Index: [Current step number]
    + Screenshot: [Current visual state]
    + Trajectory: [Previous action sequence]
    + Chosen Action: [Optimal action for this step]
    + Rejected Action Set: [Existing invalid actions to avoid]
    </Input Structure>

    <Your Task>
    Analyze the multimodal context (screenshot + trajectory) and generate ONE new rejected action that:
    + Appears superficially reasonable but contains critical flaws
    + Either: 
        a) Targets incorrect UI elements, OR 
        b) Uses improper operation sequences, OR 
        c) Performs irrelevant actions for the current step
    + Explicitly differs from all actions in the rejected action set
    </Your Task>

    <Output Format>
    Return ONLY the generated rejected action as a concise imperative statement, e.g.: 
    "Click the 'Back' button in the top-left corner"
    </Output Format>

    <Critical Constraints>
    + The action must be executable but ineffective
    + Must NOT reuse verbs/UI elements from the rejected action set
    + Must NOT combine multiple valid actions in illogical ways
    </Critical Constraints>
'''

user_prompt = """
+ Instruction: <Task description>
+ Step Index: <Current step number>
+ Trajectory: <Previous action sequence>
+ Chosen Action: <Optimal action for this step>
+ Rejected Action Set: <Existing invalid actions to avoid>
"""


def generate_action(row, rejected_action_set):
    user_message = user_prompt
    user_message = user_message.replace("<Task description>", row['instruction'])
    user_message = user_message.replace("<Current step number>", str(row['step_idx']))
    user_message = user_message.replace("<Previous action sequence>", row['reason_steps'])
    user_message = user_message.replace("<Optimal action for this step>", row['chosen'])
    user_message = user_message.replace("<Existing invalid actions to avoid>", rejected_action_set)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
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
                    "type": "image_url",
                    "image_url": {"url": row['image_url']},
                },
            ]
        }
    ]

    while True:
        response = ask_chatgpt_ant(messages)
        if (response != False):
            if response[0] == '"':
                response = response[1:-1]
            print("response = \n", response)
            print("\n")
            break

    return response



# list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"] # , "SRM_5.csv", "SRM_6.csv"
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result', 'rejected_modified']


final_csv_file = "SRM_test_ok_modify3.csv"
# if not os.path.exists(final_csv_file):
#     with open(final_csv_file, 'w', newline='') as file:
#         fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result', 'rejected_modified']
#         writer = csv.DictWriter(file, fieldnames=fields)
#         writer.writeheader()


srm_ok_csv_file = "SRM_test_ok_modify_replace.csv" # 筛掉人工标注时不合理数据的srm数据

srm_original_filter_csv_file = "SRM_test_ok_modify_replace_filter.csv" # 已经去重的srm数据+修改的数据

df_ok = pd.read_csv(srm_ok_csv_file, encoding="GB2312")

df_filter = pd.read_csv(srm_original_filter_csv_file, encoding="GB2312")


action_map = {}

for index2, row2 in df_filter.iterrows():
    id = row2['instruction'] + '_' + str(row2['step_idx']) + '_' + row2['type'] + '_' + row2['reason_steps']
    action_map[id] = []
    action_map[id].append(row2['rejected'])


with open(final_csv_file, 'w', newline='', encoding="GB2312") as file:
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected',
              'result']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for index1, row1 in df_ok.iterrows():
        print(index1)

        flag = 0
        for index2, row2 in df_filter.iterrows():

            if (row1['instruction'] == row2['instruction']) and (row1['step_idx'] == row2['step_idx']) and (row1['type'] == row2['type']) and (row1['reason_steps'] == row2['reason_steps']) and (row1['chosen'] == row2['chosen']) and (row1['rejected'] == row2['rejected']):
                # 就是这条
                if (row1['compare_id'] == row2['compare_id']) and (row1['step_idx'] == row2['step_idx']) and (row1['type'] == row2['type']): # 就是该数据
                    writer.writerow(
                        {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                         'instruction': row1['instruction'], 'type': row1['type'],
                         'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                         'chosen': row1['chosen'], 'rejected': row1['rejected'],
                         'result': row1['result']})
                    flag = 1
                    break
                else: # 是其它的重复数据
                    id = row1['instruction'] + '_' + str(row1['step_idx']) + '_' + row1['type'] + '_' + row1['reason_steps']

                    for index3, row3 in df_filter.iterrows():
                        if (row1['type'] == 'traj'):
                            if not (row3['type'] == 'traj'):
                                continue
                            else:
                                # 如果不是同一个任务
                                if not (row1['instruction'] == row3['instruction']):
                                    if (row1['rejected'] != row3['rejected']) and (row1['chosen'] != row3['rejected']) and (row3['rejected'] not in action_map[id]):
                                        writer.writerow(
                                            {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                                             'instruction': row1['instruction'], 'type': row1['type'],
                                             'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                                             'chosen': row1['chosen'], 'rejected': row3['rejected'],
                                             'result': row1['result']})
                                        flag = 1
                                        break
                                    elif (row1['rejected'] != row3['chosen']) and (row1['chosen'] != row3['chosen']) and (row3['chosen'] not in action_map[id]):
                                        writer.writerow(
                                            {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                                             'instruction': row1['instruction'], 'type': row1['type'],
                                             'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                                             'chosen': row1['chosen'], 'rejected': row3['chosen'],
                                             'result': row1['result']})
                                        flag = 1
                                        break
                        else:
                            if (row3['type'] == 'traj'):
                                continue
                            else:
                                if not (row1['instruction'] == row3['instruction']):
                                    if (row1['rejected'] != row3['rejected']) and (row1['chosen'] != row3['rejected']) and (
                                            row3['rejected'] not in action_map[id]):
                                        writer.writerow(
                                            {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                                             'instruction': row1['instruction'], 'type': row1['type'],
                                             'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                                             'chosen': row1['chosen'], 'rejected': row3['rejected'],
                                             'result': row1['result']})
                                        flag = 1
                                        break
                                    elif (row1['rejected'] != row3['chosen']) and (row1['chosen'] != row3['chosen']) and (
                                            row3['chosen'] not in action_map[id]):
                                        writer.writerow(
                                            {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                                             'instruction': row1['instruction'], 'type': row1['type'],
                                             'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                                             'chosen': row1['chosen'], 'rejected': row3['chosen'],
                                             'result': row1['result']})
                                        flag = 1
                                        break
                break

        if flag == 0:
            rejected_action_set = '{' + ',\n'.join(map(str, action_map[id])) + '}'
            new_action = generate_action(row1, rejected_action_set)
            action_map[id].append(new_action)

            writer.writerow(
                {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                 'instruction': row1['instruction'], 'type': row1['type'],
                 'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                 'chosen': row1['chosen'], 'rejected': new_action,
                 'result': row1['result']})