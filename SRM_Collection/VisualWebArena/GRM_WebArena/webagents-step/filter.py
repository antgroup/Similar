import os
import shutil

filepath = "generate_data/10.4/tasks/webarena"

list = []
for file in os.listdir(filepath):
    # print("file1 = ", file)
    try:
        files = os.listdir(filepath + '/' + file)
        # print("len(files) = ", len(files))

        num = len(files)
        if num == 0:
            print("file = ", file)
            list.append(file)
            shutil.rmtree(filepath + '/' + file)
    except:
        print("not a directory")

print("list = ", list)

## list =  ['280', '689', '286', '281', '439', '437', '656', '436', '431', '438', '657', '465', '510', '528', '521', '519', '572', '324', '586', '323', '575', '588', '385', '529', '313', '589', '574', '587', '573', '325', '508', '361', '530', '506', '794', '332', '335', '351', '507', '531', '509', '333', '795', '792', '284', '283', '277', '279', '282', '285', '655', '467', '469', '298', '434', '433', '653', '466', '654', '432', '435', '299', '571', '327', '585', '329', '387', '328', '319', '321', '386']