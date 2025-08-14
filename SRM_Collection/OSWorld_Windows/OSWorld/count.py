import os

filepath = "evaluation_examples/examples/Windows"

num_all = 0
for filedir in os.listdir(filepath):
    print("domain = ", filedir)
    now_filepath = os.path.join(filepath, filedir)
    files = os.listdir(now_filepath)
    num_files = len(files)
    print("num_files = ", num_files)
    num_all += num_files

print("num_all = ", num_all)