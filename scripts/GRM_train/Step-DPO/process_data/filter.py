import os
import pandas as pd
import csv

csv_file = "generate_data/Reward_Model2.csv"
final_csv_file = "generate_data/Reward_Model3.csv"

data = pd.read_csv(csv_file, encoding='iso-8859-1')

completed_id_dict = {}
completed_id = []
for (index, row) in data.iterrows():
    idx = int(row['task_ID'][:row['task_ID'].find('_')])
    if idx not in completed_id_dict:
        if (row['TR'] == 1) and (row['C'] == 1):
            completed_id_dict[idx] = 1
            completed_id.append(idx)

print("completed_id = ", completed_id)

with open(csv_file,'r', encoding='iso-8859-1') as load_input:
    with open(final_csv_file, 'w', newline='', encoding='iso-8859-1') as out_output:
        # fields = ['instruction', 'step_idx', 'observation_url', 'action', 'action_origin', 'IP', 'E', 'TC', 'TR', 'C', 'quality', 'annotation_reason']
        # writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()

        ereader = csv.reader(load_input)  # 用reader函数读入文件指针
        ewriter = csv.writer(out_output)  # 用writer函数读入文件指针
        eheader = next(ereader)  # 取出文件的第一行，也就是表头
        ewriter.writerow(eheader)  # 把表头写入我们要写入的文件

        # for (index, row) in data.iterrows():
        #     idx = int(row['task_ID'][:row['task_ID'].find('_')])
        #     if idx in completed_id_dict:
        #         writer.writerow(row)

        for row_list in ereader:
            # print(row_list[0])
            idx = int(row_list[0][:row_list[0].find('_')])
            if idx in completed_id_dict:
                ewriter.writerow(row_list)