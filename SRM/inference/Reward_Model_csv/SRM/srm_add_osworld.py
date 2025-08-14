import os
import csv
import pandas as pd
import json
import chardet



list = ["SRM_5.csv", "SRM_6.csv"]
fields = ['compare_id', 'step_idx', 'instruction', 'type', 'reason_steps', 'image_url', 'chosen', 'rejected', 'result']

csv_file = "SRM_test_ok_modify3.csv"
final_csv_file = "SRM_test_ok_modify3_add.csv"

df_csv = pd.read_csv(csv_file, encoding="GB2312")

for filepath in list:
    csv_osworld_file = filepath

    # with open(csv_file, 'rb') as file:
    #     character = chardet.detect(file.read())
    # print("character = ", character)

    df_now = pd.read_csv(csv_osworld_file, encoding="GB2312")
    # row_count, column_count = df_now.shape
    # print(f"行数: {row_count}, 列数: {column_count}")
    # 列名
    # print(df_now.columns)
    df_now.loc[df_now['result'] == '合理', ['result']] = 1
    df_now.loc[df_now['result'] == '不合理', ['result']] = 0

    df_now = df_now[df_now['compare_id'].str.startswith("OSWorld")]

    df_now = df_now.loc[df_now['result'] == 1]

    df_now_select = df_now[fields]
    df_csv = pd.concat([df_csv, df_now_select])

df_final = df_csv[fields]
df_final.to_csv(final_csv_file, encoding="GB2312")
