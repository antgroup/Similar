import os
import csv
import pandas as pd
import json
import chardet



list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv"]  # , "SRM_5.csv", "SRM_6.csv"
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']

csv_file = "SRM_test_ok_modify3_add.csv"
final_csv_file = "SRM_test_ok_modify3_fix.csv"

df_csv = pd.read_csv(csv_file, encoding="GB2312")


with open(final_csv_file, 'w', newline='') as file:
    fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for index, row in df_csv.iterrows():
        print(index)
        # print(row['rejected_modified'])
        if row['type'] == 'traj':
            # print(index)
            # print(row['rejected_modified'])
            writer.writerow(
                {'compare_id': row['compare_id'], 'step_idx': row['step_idx'],
                 'instruction': row['instruction'], 'type': row['type'],
                 'reason_steps': row['reason_steps'], 'image_url': row['image_url'],
                 'chosen': row['chosen'], 'rejected': row['rejected'],
                 'result': row['result'],})
        else:
            if row['chosen'].startswith('\nSTEP'):
                chosen = 'Step ' + str(row['step_idx']) + ' ' + row['chosen'][row['chosen'].find(":"):]
                # print("now = ", row['chosen'][row['chosen'].find(":"):])
            elif row['chosen'].startswith('\nStep'):
                chosen = row['chosen']
            else:
                if row['chosen'][0].isupper():
                    chosen = 'Step ' + str(row['step_idx']) + ' : ' + row['chosen'][0].lower() + row['chosen'][1:]
                else:
                    chosen = 'Step ' + str(row['step_idx']) + ' : ' + row['chosen']

            if row['rejected'].startswith('\nSTEP'):
                rejected = 'Step ' + str(row['step_idx']) + ' ' + row['rejected'][row['rejected'].find(":"):]
            elif row['rejected'].startswith('\nStep'):
                rejected = row['rejected']
            else:
                if row['rejected'][0].isupper():
                    rejected = 'Step ' + str(row['step_idx']) + ' : ' + row['rejected'][0].lower() + row['rejected'][1:]
                else:
                    rejected = 'Step ' + str(row['step_idx']) + ' : ' + row['rejected']

            writer.writerow(
                {'compare_id': row['compare_id'], 'step_idx': row['step_idx'],
                 'instruction': row['instruction'], 'type': row['type'],
                 'reason_steps': row['reason_steps'], 'image_url': row['image_url'],
                 'chosen': chosen, 'rejected': rejected,
                 'result': row['result'], })

            # break



# df_csv=df_csv[fields]

# df_csv.to_csv("SRM_test_ok_modify_replace.csv", index=False)
