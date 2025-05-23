import os
import pandas as pd
import json


def eval_grm_bench(df_examples, acc_column="correct"):
    types = [
        "IP",
        "E",
        "TC",
        "TR",
        "C",
        "total",
    ]
    df_acc = pd.DataFrame(columns=["type", "num", "accuracy"])
    for type_now in types:
        df_subset = df_examples[df_examples["type"] == type_now]
        acc = df_subset[acc_column].values.mean()
        row = {
            "type": type_now,
            "num": len(df_subset),
            "accuracy": acc,
        }
        # print("\nrow = ", row)
        df_acc = pd.concat([df_acc, pd.DataFrame(row, index=[0])], ignore_index=True)

    return df_acc

choose = [1, 2]
result = pd.read_csv("Benchmark_test_internvl2_5_1.7.csv")

# rest = []
# for index, row in result.iterrows():
#     if not (row["choose"]==1.0 or row["choose"]==2.0):
#         rest.append(row["compare_id"])
# print("rest = ", rest)
# print("\n")

result = result.loc[result["choose"].isin(choose)]
print("len(result) = ", len(result))

correct_list = []

for index, row in result.iterrows():
    if row["choose"] == 1:
        correct_list.append(1.0)
    else:
        correct_list.append(0.0)

result['correct'] = [score for score in correct_list]
acc_per_type = eval_grm_bench(result)
print("\nGRMBench Scores:")
print(acc_per_type)

