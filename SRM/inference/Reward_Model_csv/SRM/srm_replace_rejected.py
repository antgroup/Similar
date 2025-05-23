import os
import csv
import pandas as pd
import json
import chardet



list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"]  # , "SRM_5.csv", "SRM_6.csv"
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']

csv_file = "SRM_test_ok_modify.csv"

final_csv_file = "SRM_test_ok_modify_replace.csv"

# df_csv = pd.read_csv(csv_file)
df_csv = pd.read_csv(csv_file, encoding="GB2312")

# def get_rejected(x):
#     if pd.isna(x['rejected_modified']):
#         return x['rejected']
#     else:
#         return x['rejected_modified']
#
# df_csv.loc[:, 'rejected1'] = df_csv.apply(get_rejected, axis=1)

if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for index, row in df_csv.iterrows():
            # print(index)
            # print(row['rejected_modified'])
            if not pd.isna(row['rejected_modified']):
                # print(index)
                # print(row['rejected_modified'])
                writer.writerow(
                    {'compare_id': row['compare_id'], 'step_idx': row['step_idx'],
                     'instruction': row['instruction'], 'type': row['type'],
                     'reason_steps': row['reason_steps'], 'image_url': row['image_url'],
                     'chosen': row['chosen'], 'rejected': row['rejected_modified'],
                     'result': row['result'],})
            else:
                writer.writerow(
                    {'compare_id': row['compare_id'], 'step_idx': row['step_idx'],
                     'instruction': row['instruction'], 'type': row['type'],
                     'reason_steps': row['reason_steps'], 'image_url': row['image_url'],
                     'chosen': row['chosen'], 'rejected': row['rejected'],
                     'result': row['result'], })


# df_csv.loc[df_csv['rejected_modified'] == 'None', ['rejected']] = df_csv['rejected']
# df_csv.loc[df_csv['rejected_modified'] != 'None', ['rejected']] = df_csv['rejected_modified']

# df_csv=df_csv[fields]

# df_csv.to_csv("SRM_test_ok_modify_replace.csv", index=False)
