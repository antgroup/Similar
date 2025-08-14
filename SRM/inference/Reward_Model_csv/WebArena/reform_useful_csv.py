import os
import pandas as pd
import csv
import json

csv_file = "Reward_Model_WebArena.csv"
csv_useful_file = "Reward_Model_WebArena_useful.csv"
csv_useful_train_file = "Reward_Model_WebArena_useful_train.csv"

task_id_useful_list =  [3, 22, 23, 24, 28, 29, 30, 41, 42, 43, 44, 69, 79, 94, 95, 115, 118, 128, 129, 130, 132, 134, 135, 149, 150, 156, 158, 160, 161, 164, 166, 167, 169, 177, 183, 188, 190, 198, 199, 202, 203, 205, 208, 209, 211, 212, 225, 227, 230, 231, 247, 258, 260, 261, 262, 264, 273, 274, 275, 276, 278, 302, 303, 305, 306, 308, 310, 311, 312, 313, 314, 315, 317, 318, 322, 326, 340, 348, 351, 354, 355, 358, 359, 360, 361, 368, 376, 384, 387, 388, 392, 395, 397, 447, 465, 468, 472, 474, 477, 478, 479, 491, 511, 512, 514, 515, 516, 517, 518, 533, 535, 539, 580, 581, 650, 651, 652, 691, 692, 693, 723, 726, 731, 753, 754, 772, 775, 784, 785, 787, 790, 793, 794, 795]
task_id_useful_test_list =  [491, 258, 794, 652, 274, 130, 731, 211, 479, 161, 190, 322, 693, 41, 132, 29, 384, 183, 517, 150, 149, 69, 355, 95, 387, 305]
subtask_id_useful_list = []
subtask_id_useful_train_list = []

data = pd.read_csv(csv_file, encoding="iso-8859-1")

for index, row in data.iterrows():
    subtask_id = row['task_ID']
    task_id = int(subtask_id.split('_')[0])
    if (task_id in task_id_useful_list):
        subtask_id_useful_list.append(subtask_id)
        if (task_id not in task_id_useful_test_list):
            subtask_id_useful_train_list.append(subtask_id)

csv_useful_data = data.loc[data['task_ID'].isin(subtask_id_useful_list)]
csv_useful_data.to_csv(csv_useful_file, index=False)

csv_useful_train_data = data.loc[data['task_ID'].isin(subtask_id_useful_train_list)]
csv_useful_train_data.to_csv(csv_useful_train_file, index=False)



