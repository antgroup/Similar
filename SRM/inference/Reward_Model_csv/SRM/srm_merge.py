import os
import csv
import pandas as pd
import json
import chardet

def get_statistic(df_examples):
    types = [
        "IP",
        "E",
        "TC",
        "TR",
        "C",
        "total",
        "traj",
    ]
    df_st = pd.DataFrame(columns=["type", "num"])
    for type_now in types:
        df_subset = df_examples[df_examples["type"] == type_now]
        row = {
            "type": type_now,
            "num": len(df_subset),
        }
        df_st = pd.concat([df_st, pd.DataFrame(row, index=[0])], ignore_index=True)

    row = {
        "type": 'all',
        "num": len(df_examples),
    }
    df_st = pd.concat([df_st, pd.DataFrame(row, index=[0])], ignore_index=True)

    return df_st


list = ["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv", "SRM_5.csv", "SRM_6.csv"] #
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']


final_csv_file = "SRM_test_original.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_csv_file_2 = "SRM_test_ok.csv"
if not os.path.exists(final_csv_file_2):
    with open(final_csv_file_2, 'w', newline='') as file:
        fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

df_final = pd.read_csv(final_csv_file)

df_final = df_final.loc[df_final['result'] == 1]

df_final.to_csv(final_csv_file_2)

df_st = get_statistic(df_final)

print(df_st)

# df_count = pd.DataFrame(columns=['counts'], index=["SRM_1.csv", "SRM_2.csv", "SRM_3.csv", "SRM_4.csv", "SRM_5.csv", "SRM_6.csv"])

# for filepath in list:
#     csv_file = filepath
#
#     # with open(csv_file, 'rb') as file:
#     #     character = chardet.detect(file.read())
#     # print("character = ", character)
#
#     df_now = pd.read_csv(csv_file, encoding="GB2312")
#     row_count, column_count = df_now.shape
#     # print(f"行数: {row_count}, 列数: {column_count}")
#     # 列名
#     # print(df_now.columns)
#     df_now.loc[df_now['result'] == '合理', ['result']] = 1
#     df_now.loc[df_now['result'] == '不合理', ['result']] = 0
#
#     df_now_select = df_now[fields]
#     df_final = pd.concat([df_final, df_now_select])
#     df_st = get_statistic(df_now)
#     print(filepath)
#     print(df_st)
#     print("\n")

# df_final = df_final[fields]
# df_final.to_csv(final_csv_file)