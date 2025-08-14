import os
import json
import csv
import requests
import shutil
import pickle

url = "https://api.superbed.cn/upload"

# 通过链接上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057", "src": "https://ww1.sinaimg.cn/large/005YhI8igy1fv09liyz9nj30qo0hsn0e"})

# 通过文件上传
# resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open("demo.jpg", "rb")})
# print(resp.json())

filepath = "generate_data/" # 标注数据文件夹
task_filepath = "./checkpoint" # task文件夹

num_useless = 0
num_already = 0
num_empty = 0
num_no = 0
num_have = 0
num_lack = 0
delete_list = []

dict =  {'ExpenseAddSingle': 1, 'MarkorDeleteNote': 2, 'RecipeAddSingleRecipe': 3, 'TasksIncompleteTasksOnDate': 4, 'SimpleCalendarAddOneEventInTwoWeeks': 5, 'SystemBluetoothTurnOn': 6, 'ClockTimerEntry': 7, 'RecipeDeleteSingleWithRecipeWithNoise': 8, 'SimpleCalendarAddOneEvent': 9, 'SimpleCalendarAnyEventsOnDate': 10, 'RecipeAddMultipleRecipes': 11, 'TasksDueOnDate': 12, 'MarkorCreateNote': 13, 'RecipeDeleteMultipleRecipesWithConstraint': 14, 'ExpenseAddMultiple': 15, 'SystemWifiTurnOffVerify': 16, 'RecipeDeleteMultipleRecipesWithNoise': 17, 'SimpleSmsReplyMostRecent': 18, 'RecipeDeleteSingleRecipe': 19, 'ExpenseDeleteMultiple': 20, 'SimpleCalendarDeleteEvents': 21, 'VlcCreatePlaylist': 22, 'SimpleSmsReply': 23, 'CameraTakeVideo': 24, 'RecipeAddMultipleRecipesFromImage': 25, 'ExpenseDeleteSingle': 26, 'MarkorEditNote': 27, 'RecipeDeleteMultipleRecipes': 28, 'SystemBrightnessMaxVerify': 29, 'NotesIsTodo': 30, 'TasksHighPriorityTasks': 31, 'MarkorCreateNoteAndSms': 32, 'MarkorDeleteAllNotes': 33, 'BrowserDraw': 34, 'MarkorMoveNote': 35, 'SystemWifiTurnOnVerify': 36, 'TurnOffWifiAndTurnOnBluetooth': 37, 'AudioRecorderRecordAudio': 38, 'SimpleCalendarAddOneEventRelativeDay': 39, 'ExpenseAddMultipleFromMarkor': 40, 'FilesMoveFile': 41, 'MarkorCreateFolder': 42, 'SystemCopyToClipboard': 43, 'BrowserMaze': 44, 'ExpenseDeleteDuplicates': 45, 'SimpleDrawProCreateDrawing': 46, 'SimpleSmsSendReceivedAddress': 47, 'VlcCreateTwoPlaylists': 48, 'SystemBluetoothTurnOnVerify': 49, 'SystemWifiTurnOff': 50, 'RecipeAddMultipleRecipesFromMarkor': 51, 'TurnOnWifiAndOpenApp': 52, 'ContactsNewContactDraft': 53, 'SystemWifiTurnOn': 54, 'SystemBluetoothTurnOffVerify': 55, 'SportsTrackerActivityDuration': 56, 'NotesMeetingAttendeeCount': 57, 'SimpleCalendarAddRepeatingEvent': 58, 'ExpenseDeleteDuplicates2': 59, 'SimpleCalendarEventOnDateAtTime': 60, 'SimpleSmsSend': 61, 'NotesTodoItemCount': 62, 'ClockStopWatchPausedVerify': 63, 'FilesDeleteFile': 64, 'NotesRecipeIngredientCount': 65, 'ContactsAddContact': 66, 'MarkorMergeNotes': 67, 'RecipeAddMultipleRecipesFromMarkor2': 68, 'RecipeDeleteDuplicateRecipes': 69, 'SaveCopyOfReceiptTaskEval': 70, 'SimpleCalendarLocationOfEvent': 71, 'SimpleCalendarFirstEventAfterStartTime': 72, 'SystemBrightnessMax': 73, 'MarkorDeleteNewestNote': 74, 'SportsTrackerTotalDistanceForCategoryOverInterval': 75, 'SystemBrightnessMinVerify': 76, 'CameraTakePhoto': 77, 'AudioRecorderRecordAudioWithFileName': 78, 'SystemBrightnessMin': 79, 'OsmAndFavorite': 80, 'SimpleCalendarDeleteEventsOnRelativeDay': 81, 'SimpleCalendarEventsInTimeRange': 82, 'MarkorChangeNoteContent': 83, 'MarkorTranscribeReceipt': 84, 'SimpleCalendarEventsInNextWeek': 85, 'SportsTrackerLongestDistanceActivity': 86, 'SystemBluetoothTurnOff': 87, 'TasksDueNextWeek': 88, 'SimpleSmsResend': 89, 'SimpleCalendarAddOneEventTomorrow': 90, 'OsmAndMarker': 91, 'SportsTrackerActivitiesCountForWeek': 92, 'ExpenseDeleteMultiple2': 93, 'RecipeDeleteDuplicateRecipes3': 94, 'RecipeDeleteDuplicateRecipes2': 95, 'SimpleCalendarNextEvent': 96, 'OsmAndTrack': 97, 'TasksHighPriorityTasksDueOnDate': 98, 'OpenAppTaskEval': 99, 'MarkorAddNoteHeader': 100, 'SportsTrackerTotalDurationForCategoryThisWeek': 101, 'SportsTrackerActivitiesOnDate': 102, 'BrowserMultiply': 103, 'ClockStopWatchRunning': 104, 'TasksCompletedTasksForDate': 105, 'ExpenseAddMultipleFromGallery': 106, 'MarkorCreateNoteFromClipboard': 107, 'SimpleCalendarNextMeetingWithPerson': 108, 'SimpleSmsSendClipboardContent': 109, 'MarkorTranscribeVideo': 110}
dict_idx = 0

for filedir in os.listdir(filepath):
    # print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if filedir[-2] == '-':
        task_id = filedir[:filedir.find('-')]
    else:
        task_id = filedir

    print("task_id = ", task_id)

    instruction = ''
    for filedir2 in os.listdir(task_filepath):
        if filedir2[:-6] == task_id:
            path = os.path.join(task_filepath, filedir2)
            # print("path = ", path)
            f = open(path, 'rb')
            data = pickle.load(f)

            instruction = data[0]["goal"]

            break

    # print("instruction = ", instruction)
    if instruction == '':
        break

    idx = 0
    if task_id not in dict:
        dict_idx = dict_idx + 1
        dict[task_id] = dict_idx
        idx = dict_idx
    else:
        idx = dict[task_id]

    print("idx = ", idx)

    # 存入到csv文件中
    if filedir[-2] == '-':
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + filedir[filedir.find('-') + 1 : ] + ".csv"
    else:
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + ".csv"
    # print("csv_file = ", csv_file)

    if os.path.exists(csv_file):
        # print("csv_file = ", csv_file)
        if len(open(csv_file).readlines()) <= 1:
            num_useless += 1
            os.remove(csv_file)
            # continue

    if os.path.exists(csv_file):
        num_already += 1
        continue


    evaluation_score_file = os.path.join(filepath, filedir, "evaluation_score.json") # 步骤级多维度评分文件
    if os.path.exists(evaluation_score_file): # 该文件数据有效

        print("\nnow = ", filedir)
        print("csv_file = ", csv_file)
        print("\n")

        with open(csv_file, 'w', newline='') as file:
            fields = ['instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            with open(evaluation_score_file, 'r', encoding='UTF-8') as f:
                evaluation_score_dict = json.load(f)

            i = 1
            for (key, value) in evaluation_score_dict.items():
                # try:
                img_ori = os.path.join(filepath, task_id) + '/webpage' + str(key) + '.png'
                img = os.path.join(filepath, filedir) + '/webpage' + str(i) + '.png'
                if not os.path.exists(img_ori):
                    break
                if not os.path.exists(img):
                    num_lack += 1
                    # break
                    shutil.copyfile(img_ori, img)
                i += 1

            pic_num = 0
            for (key, value) in evaluation_score_dict.items():
                img = os.path.join(filepath, filedir) + '/webpage' + str(key) + '.png'
                if os.path.exists(img):
                    pic_num = 1
                    # continue
                else:
                    flag = 1
                    # continue

                try:
                    resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"}, files={"file": open(img, "rb")})
                    print("resp.json() = ", resp.json())
                    observation_url = resp.json()['url']

                    writer.writerow(
                        {'instruction': instruction, 'step_idx': int(value['step_idx']), 'observation_url': observation_url,
                         'action': str(value['action']), 'IP': float(value['IP']), 'E': float(value['E']),
                         'TC': float(value['TC']), 'TR': float(value['TR']), 'C': float(value['C'])})
                except Exception as e:
                    # continue
                    break

            # if pic_num:
            #     num_have += 1
            #     print("\nthere are images")
            #
            #     i = 1
            #     for (key, value) in evaluation_score_dict.items():
            #         # try:
            #         img_ori = os.path.join(filepath, idx) + '/webpage' + str(key) + '.png'
            #         img = os.path.join(filepath, filedir) + '/webpage' + str(i) + '.png'
            #         if not os.path.exists(img):
            #             num_lack += 1
            #             break
            #             # shutil.copyfile(img_ori, img)
            #         i += 1
            # else:
            #     num_no += 1
            #     print("\nthere no images")
                # path = os.path.join(filepath, filedir)
                # shutil.rmtree(path)
    else: # 该文件夹数据无效
        num_empty += 1
        continue

print("\ndict = ", dict)
print("\n")
print("num_already = ", num_already)
print("num_empty = ", num_empty)
print("num_have = ", num_have)
print("num_lack = ", num_lack)
print("num_no = ", num_no)
print("num_useless = ", num_useless)
