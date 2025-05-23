import os
import pandas as pd
import chardet

# csv_file = "Annotation_Result/【iTAG】正式1_Generalist_Reward_Model数据标注_2024101800104737240_GBK__20241028040911.csv"
final_csv_file = "generate_data/Reward_Model.csv"
# with open(final_csv_file, 'w', newline='') as file:
#     fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C', '任务质量判别', '任务质量为0，标注原因']
#     writer = csv.DictWriter(file, fieldnames=fields)
#     writer.writeheader()

# with open(final_csv_file, 'rb') as f:
#     result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测
#
# print(result['encoding'])  # 打印检测到的编码

final_data = pd.read_csv(final_csv_file, encoding='iso-8859-1')
final_data = final_data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C', 'quality', 'annotation_reason']]
final_data.to_csv("generate_data/Reward_Model2.csv")