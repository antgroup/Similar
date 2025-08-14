import os
import json
import shutil

filepath = "generate_data"

done = []
to_do = []
without_all_length = []

for filedir in os.listdir(filepath):
    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if not filedir[-2] == '-':
        print("\nnow = ", filedir)
        action_list_file = os.path.join(filepath, filedir, "action_list.json")

        if not os.path.exists(action_list_file):
            to_do.append(filedir)
            # shutil.rmtree(os.path.join(filepath, filedir))
        else:
            with open(action_list_file, 'r', encoding='UTF-8') as f:
                action_list = json.load(f)

            all_length_file = os.path.join(filepath, filedir, "all_length.text")

            if not os.path.exists(all_length_file):
                without_all_length.append(filedir)
                to_do.append(filedir)
                continue

            with open(all_length_file, "r", encoding='utf-8') as f:
                all_length = float(f.readline())

            if all_length == 0.0:
                print("\n000000")
                done.append(filedir)
                # if not os.path.exists(all_length_file):
                #     all_length = 0.0
                #     with open(all_length_file, 'w') as f:
                #         f.write(str(all_length))
            else:
                print("\n111111")
                evaluation_score_file = os.path.join(filepath, filedir, "evaluation_score.json")
                if os.path.exists(evaluation_score_file):
                    done.append(filedir)

                    # with open(evaluation_score_file, 'r', encoding='UTF-8') as f:
                    #     evaluation_score_dict = json.load(f)
                    #
                    # flag = 0
                    # for (key, value) in evaluation_score_dict.items():
                    #     if ('stop' in value['action']):
                    #         print("\nthis task is done.")
                    #         done.append(filedir)
                    #         flag = 1
                    #         break
                    # if flag == 0:
                    #     print('\nthis task is not be fully completed.')
                    #     to_do.append(filedir)
                else:
                    to_do.append(filedir)


done.sort()
to_do.sort()
without_all_length.sort()

print("\ndone = ", done)
print("num done = %d\n" % len(done))

print("\nto_do = ", to_do)
print("num to do = %d\n" % len(to_do))

print("\nwithout_all_length = ", without_all_length)
print("num without_all_length = %d\n" % len(without_all_length))


