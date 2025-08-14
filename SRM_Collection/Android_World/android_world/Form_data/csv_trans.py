import os
import csv
import pandas as pd

filepath = "generate_data" # 标注数据文件夹

final_csv_file = filepath + "/" + "Reward_Model.csv"
if not os.path.exists(final_csv_file):
    with open(final_csv_file, 'w', newline='') as file:
        # fields = ['task_ID', 'sub_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        fields = ['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

final_data = pd.read_csv(final_csv_file)
# print("final_data = \n", final_data)

dict =  {'ExpenseAddSingle': 1, 'MarkorDeleteNote': 2, 'RecipeAddSingleRecipe': 3, 'TasksIncompleteTasksOnDate': 4, 'SimpleCalendarAddOneEventInTwoWeeks': 5, 'SystemBluetoothTurnOn': 6, 'ClockTimerEntry': 7, 'RecipeDeleteSingleWithRecipeWithNoise': 8, 'SimpleCalendarAddOneEvent': 9, 'SimpleCalendarAnyEventsOnDate': 10, 'RecipeAddMultipleRecipes': 11, 'TasksDueOnDate': 12, 'MarkorCreateNote': 13, 'RecipeDeleteMultipleRecipesWithConstraint': 14, 'ExpenseAddMultiple': 15, 'SystemWifiTurnOffVerify': 16, 'RecipeDeleteMultipleRecipesWithNoise': 17, 'SimpleSmsReplyMostRecent': 18, 'RecipeDeleteSingleRecipe': 19, 'ExpenseDeleteMultiple': 20, 'SimpleCalendarDeleteEvents': 21, 'VlcCreatePlaylist': 22, 'SimpleSmsReply': 23, 'CameraTakeVideo': 24, 'RecipeAddMultipleRecipesFromImage': 25, 'ExpenseDeleteSingle': 26, 'MarkorEditNote': 27, 'RecipeDeleteMultipleRecipes': 28, 'SystemBrightnessMaxVerify': 29, 'NotesIsTodo': 30, 'TasksHighPriorityTasks': 31, 'MarkorCreateNoteAndSms': 32, 'MarkorDeleteAllNotes': 33, 'BrowserDraw': 34, 'MarkorMoveNote': 35, 'SystemWifiTurnOnVerify': 36, 'TurnOffWifiAndTurnOnBluetooth': 37, 'AudioRecorderRecordAudio': 38, 'SimpleCalendarAddOneEventRelativeDay': 39, 'ExpenseAddMultipleFromMarkor': 40, 'FilesMoveFile': 41, 'MarkorCreateFolder': 42, 'SystemCopyToClipboard': 43, 'BrowserMaze': 44, 'ExpenseDeleteDuplicates': 45, 'SimpleDrawProCreateDrawing': 46, 'SimpleSmsSendReceivedAddress': 47, 'VlcCreateTwoPlaylists': 48, 'SystemBluetoothTurnOnVerify': 49, 'SystemWifiTurnOff': 50, 'RecipeAddMultipleRecipesFromMarkor': 51, 'TurnOnWifiAndOpenApp': 52, 'ContactsNewContactDraft': 53, 'SystemWifiTurnOn': 54, 'SystemBluetoothTurnOffVerify': 55, 'SportsTrackerActivityDuration': 56, 'NotesMeetingAttendeeCount': 57, 'SimpleCalendarAddRepeatingEvent': 58, 'ExpenseDeleteDuplicates2': 59, 'SimpleCalendarEventOnDateAtTime': 60, 'SimpleSmsSend': 61, 'NotesTodoItemCount': 62, 'ClockStopWatchPausedVerify': 63, 'FilesDeleteFile': 64, 'NotesRecipeIngredientCount': 65, 'ContactsAddContact': 66, 'MarkorMergeNotes': 67, 'RecipeAddMultipleRecipesFromMarkor2': 68, 'RecipeDeleteDuplicateRecipes': 69, 'SaveCopyOfReceiptTaskEval': 70, 'SimpleCalendarLocationOfEvent': 71, 'SimpleCalendarFirstEventAfterStartTime': 72, 'SystemBrightnessMax': 73, 'MarkorDeleteNewestNote': 74, 'SportsTrackerTotalDistanceForCategoryOverInterval': 75, 'SystemBrightnessMinVerify': 76, 'CameraTakePhoto': 77, 'AudioRecorderRecordAudioWithFileName': 78, 'SystemBrightnessMin': 79, 'OsmAndFavorite': 80, 'SimpleCalendarDeleteEventsOnRelativeDay': 81, 'SimpleCalendarEventsInTimeRange': 82, 'MarkorChangeNoteContent': 83, 'MarkorTranscribeReceipt': 84, 'SimpleCalendarEventsInNextWeek': 85, 'SportsTrackerLongestDistanceActivity': 86, 'SystemBluetoothTurnOff': 87, 'TasksDueNextWeek': 88, 'SimpleSmsResend': 89, 'SimpleCalendarAddOneEventTomorrow': 90, 'OsmAndMarker': 91, 'SportsTrackerActivitiesCountForWeek': 92, 'ExpenseDeleteMultiple2': 93, 'RecipeDeleteDuplicateRecipes3': 94, 'RecipeDeleteDuplicateRecipes2': 95, 'SimpleCalendarNextEvent': 96, 'OsmAndTrack': 97, 'TasksHighPriorityTasksDueOnDate': 98, 'OpenAppTaskEval': 99, 'MarkorAddNoteHeader': 100, 'SportsTrackerTotalDurationForCategoryThisWeek': 101, 'SportsTrackerActivitiesOnDate': 102, 'BrowserMultiply': 103, 'ClockStopWatchRunning': 104, 'TasksCompletedTasksForDate': 105, 'ExpenseAddMultipleFromGallery': 106, 'MarkorCreateNoteFromClipboard': 107, 'SimpleCalendarNextMeetingWithPerson': 108, 'SimpleSmsSendClipboardContent': 109, 'MarkorTranscribeVideo': 110}
count_dict = {}
num = 0

for filedir in os.listdir(filepath):
    print("\nnow = ", filedir)

    if not os.path.isdir(os.path.join(filepath, filedir)):
        continue

    if filedir[-2] == '-':
        task_id = filedir[:filedir.find('-')]
    else:
        task_id = filedir

    idx = dict[task_id]

    if idx in count_dict:
        count_dict[idx] += 1
    else:
        count_dict[idx] = 1
    # print("idx = ", idx)

    if filedir[-2] == '-':
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + filedir[filedir.find('-') + 1 : ] + ".csv"
    else:
        csv_file = os.path.join(filepath, filedir) + "/" + str(idx) + ".csv"

    print("csv_file = ", csv_file)
    if not os.path.exists(csv_file):
        num += 1
        continue

    # if len(open(csv_file).readlines()) <= 1:
    #     print("\nnow = ", filedir)
    #     num += 1
    #     os.remove(csv_file)
    #     continue

    # if (idx == 22):
    # print("\nnow = ", filedir)
    # print("count_dict[%d] = %d" % (idx, count_dict[idx]))

    data = pd.read_csv(csv_file)
    task_ID = []
    # sub_ID = []
    for index, row in data.iterrows():
        task_ID.append('Android_World_' + str(idx) + '_' + str(count_dict[idx]))
        # sub_ID.append(count_dict[idx])
    data['task_ID'] = task_ID
    # data['sub_ID'] = sub_ID
    data = data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
    # print("data = \n", data)
    final_data = pd.concat([final_data, data])
    # print("final_data = \n", final_data)

# print("\nnum = ", num)
final_data = final_data[['task_ID', 'instruction', 'step_idx', 'observation_url', 'action', 'IP', 'E', 'TC', 'TR', 'C']]
final_data.to_csv(final_csv_file)

