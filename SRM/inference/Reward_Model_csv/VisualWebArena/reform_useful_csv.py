import os
import pandas as pd
import csv
import json

csv_file = "Reward_Model_VisualWebArena_original.csv"
csv_useful_file = "Reward_Model_VisualWebArena_useful.csv"
csv_useful_train_file = "Reward_Model_VisualWebArena_useful_train.csv"

task_id_useful_list =  ['classifieds_10', 'classifieds_106', 'classifieds_11', 'classifieds_111', 'classifieds_118', 'classifieds_120', 'classifieds_121', 'classifieds_124', 'classifieds_125', 'classifieds_127', 'classifieds_129', 'classifieds_130', 'classifieds_135', 'classifieds_140', 'classifieds_15', 'classifieds_153', 'classifieds_164', 'classifieds_167', 'classifieds_173', 'classifieds_174', 'classifieds_189', 'classifieds_194', 'classifieds_195', 'classifieds_196', 'classifieds_199', 'classifieds_209', 'classifieds_220', 'classifieds_24', 'classifieds_29', 'classifieds_49', 'classifieds_50', 'classifieds_53', 'classifieds_98', 'reddit_120', 'reddit_155', 'reddit_160', 'reddit_162', 'reddit_173', 'reddit_182', 'reddit_19', 'reddit_196', 'reddit_31', 'reddit_36', 'reddit_39', 'reddit_69', 'reddit_7', 'reddit_92', 'shopping_111', 'shopping_128', 'shopping_129', 'shopping_14', 'shopping_147', 'shopping_148', 'shopping_161', 'shopping_167', 'shopping_188', 'shopping_200', 'shopping_204', 'shopping_209', 'shopping_212', 'shopping_217', 'shopping_58', 'shopping_69', 'shopping_70', 'shopping_71', 'shopping_73', 'shopping_80', 'shopping_81', 'shopping_91']
task_id_useful_test_list =  ['shopping_129', 'classifieds_125', 'classifieds_173', 'shopping_204', 'classifieds_49', 'classifieds_127', 'classifieds_53', 'shopping_111', 'reddit_182', 'shopping_217', 'shopping_212', 'reddit_162', 'classifieds_153']
subtask_id_useful_list = []
subtask_id_useful_train_list = []

data = pd.read_csv(csv_file, encoding="iso-8859-1")

for index, row in data.iterrows():
    subtask_id = row['task_ID']
    if "-" in subtask_id:
        task_id = subtask_id[subtask_id.find("VisualWebArena_") + len("VisualWebArena_"): subtask_id.find("-")]
    else:
        task_id = subtask_id[subtask_id.find("VisualWebArena_") + len("VisualWebArena_"):]

    if (task_id in task_id_useful_list):
        subtask_id_useful_list.append(subtask_id)
        if (task_id not in task_id_useful_test_list):
            subtask_id_useful_train_list.append(subtask_id)

csv_useful_data = data.loc[data['task_ID'].isin(subtask_id_useful_list)]
csv_useful_data.to_csv(csv_useful_file, index=False)

csv_useful_train_data = data.loc[data['task_ID'].isin(subtask_id_useful_train_list)]
csv_useful_train_data.to_csv(csv_useful_train_file, index=False)



