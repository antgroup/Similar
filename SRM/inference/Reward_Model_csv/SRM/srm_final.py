import os
import csv
import pandas as pd
import json
import chardet


list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"]  # , "SRM_5.csv", "SRM_6.csv"
fields = ['id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected']

csv_file = "SRM_test_ok_modify3_fix2.csv"

final_csv_file = "SRM_test_final_3.4.csv"

# with open(csv_file, 'rb') as file:
#     character = chardet.detect(file.read())
# print("character = ", character)

df_csv = pd.read_csv(csv_file)

# df_csv = df_csv.iloc[:32612, :]

with open(final_csv_file, 'w', newline='', encoding="GB2312") as file:
    fields = ['id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    
    for index, row in df_csv.iterrows():
        if index>= 32612:
            break
        print(index)
        id = row['compare_id'] + '_' + str(row['step_idx']) + '_' + row['type']
        writer.writerow(
            {'id': id, 'step_idx': row['step_idx'],
             'instruction': row['instruction'], 'type': row['type'],
             'reason_steps': row['reason_steps'], 'image_url': row['image_url'],
             'chosen': row['chosen'], 'rejected': row['rejected']})

# df_csv.to_csv(final_csv_file, index=False, encoding="GB2312")