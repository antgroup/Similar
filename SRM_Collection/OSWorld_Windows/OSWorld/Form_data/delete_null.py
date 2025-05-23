import os
import csv
import pandas as pd

filepath = "generate_data/10.8/tasks/webarena" # 标注数据文件夹

for filedir in os.listdir(filepath):
    print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if '-' in filedir:
        idx = int(filedir[:filedir.find('-')])
    else:
        idx = int(filedir)

    csv_file = os.path.join(filepath, filedir) + "/" + filedir + ".csv"
    csv_file_post = os.path.join(filepath, filedir) + "/" + filedir + "_post.csv"

    if not os.path.exists(csv_file):
        continue

    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        num_null = 0
        for row in csv_reader:
            for cell in row:
                if cell == '':
                    num_null += 1
        print("num_null = ", num_null)
        if (num_null):
            print("------------------------------------------------------------!!!")

    data = pd.read_csv(csv_file)
    data = data.dropna()
    data.to_csv(csv_file_post, index=False)

    csv_file_post = os.path.join(filepath, filedir) + "/" + filedir + "_post.csv"

    with open(csv_file_post, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        num_null = 0
        for row in csv_reader:
            for cell in row:
                if cell == '':
                    num_null += 1

    print("num_null = ", num_null)