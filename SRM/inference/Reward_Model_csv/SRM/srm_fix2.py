import os
import csv
import pandas as pd
import json
import chardet


list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"]  # , "SRM_5.csv", "SRM_6.csv"
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']

csv_file = "SRM_test_ok_modify3_fix.csv"

final_csv_file = "SRM_test_ok_modify3_fix2.csv"

df_csv = pd.read_csv(csv_file, encoding="GB2312")

num = 0

compare_id_list = []
id_map = {}

with open(final_csv_file, 'w', newline='') as file:
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for index1, row1 in df_csv.iterrows():
        print(index1)
        id1 = row1['compare_id'] + '_' + str(row1['step_idx']) + '_' + row1['type']
        if id1 in id_map:
            continue
        else:
            writer.writerow(
                {'compare_id': row1['compare_id'], 'step_idx': row1['step_idx'],
                 'instruction': row1['instruction'], 'type': row1['type'],
                 'reason_steps': row1['reason_steps'], 'image_url': row1['image_url'],
                 'chosen': row1['chosen'], 'rejected': row1['rejected'],
                 'result': row1['result'], })
        for index2, row2 in df_csv.iterrows():
            if index1 >= index2:
                continue

            id2 = row2['compare_id'] + '_' + str(row2['step_idx']) + '_' + row2['type']

            if id2 in id_map:
                continue

            if (row1['instruction'] == row2['instruction']) and (row1['reason_steps'] == row2['reason_steps']) and (row1['step_idx'] == row2['step_idx']) and (row1['type'] == row2['type']) and (row1['chosen'] == row2['rejected']) and (row1['rejected'] == row2['chosen']):
                id_map[str(id2)] = 1

                print(id1 + '  ' + id2 + '\n')
                temp_map = {}
                temp_map[str(id1)] = id2
                compare_id_list.append(temp_map)
                num += 1

print("num = ", num)
print(compare_id_list)
print(id_map)

with open("modify3_fix_compare_id_list2.json", "w") as file:
    json.dump(compare_id_list, file)

with open("modify3_fix_id_map.json", "w") as file:
    json.dump(id_map, file)