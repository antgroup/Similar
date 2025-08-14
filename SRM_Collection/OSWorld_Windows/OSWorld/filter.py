import os

filepath = "evaluation_examples/examples/Windows"


for filedir in os.listdir(filepath):
    list = []
    print("\nfiledir = ", filedir)
    try:
        for file in os.listdir(os.path.join(filepath, filedir)):
            list.append(file)
    except Exception as e:
        continue

    for item in list:
        print('"' + item[:-5] + '"' + ',')