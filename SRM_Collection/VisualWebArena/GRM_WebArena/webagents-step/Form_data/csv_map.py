import os
import json
import csv
import pandas as pd
import numpy as np

filepath = "generate_data/tasks/vwa/test_reddit" # 标注数据文件夹

origin_csv_file = filepath + "/" + "Reward_Model.csv"
final_csv_file = filepath + "/" + "Reward_Model_map.csv"
# if not os.path.exists(final_csv_file):
#     with open(final_csv_file, 'w', newline='') as file:
#         fields = ['task_ID', 'sub_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
#         writer = csv.DictWriter(file, fieldnames=fields)
#         writer.writeheader()
#
# final_data = pd.read_csv(final_csv_file)

data = pd.read_csv(origin_csv_file)
# print(data)


data.loc[data['IP']>=1, 'IP'] = 3
data.loc[(data['IP']<1) & (data['IP']>=2/3), 'IP'] = 2
data.loc[(data['IP']<2/3) & (data['IP']>=1/3), 'IP'] = 1
data.loc[data['IP']<1/3, 'IP'] = 0

# data.loc[data['TC']>0, 'TC'] = 1
# data.loc[data['TC']<0, 'TC'] = -1

data.loc[data['TC']>=1/5, 'TC'] = 3
data.loc[(data['TC']<1/5) & (data['TC']>=1/10), 'TC'] = 2
data.loc[(data['TC']<1/10) & (data['TC']>=0), 'TC'] = 1
data.loc[(data['TC']==0), 'TC'] = 0
data.loc[(data['TC']<0) & (data['TC']>=-1/10), 'TC'] = -1
data.loc[(data['TC']<-1/10) & (data['TC']>=-1/5), 'TC'] = -2
data.loc[data['TC']<-1/5, 'TC'] = -3

data.loc[data['E']>=1/3, 'E'] = 5
data.loc[(data['E']<1/3) & (data['E']>=1/6), 'E'] = 4
data.loc[(data['E']<1/6) & (data['E']>=1/9), 'E'] = 3
data.loc[(data['E']<1/9) & (data['E']>=1/12), 'E'] = 2
data.loc[data['E']<1/12, 'E'] = 1


# print(data)
data = data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
data.to_csv(final_csv_file)