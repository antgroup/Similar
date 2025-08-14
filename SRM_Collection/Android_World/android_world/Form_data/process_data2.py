import os
import json
import shutil

filepath = "checkpoint"

to_do =  ['ExpenseDeleteSingle', 'MarkorTranscribeVideo', 'NotesRecipeIngredientCount', 'OsmAndTrack', 'RecipeDeleteSingleRecipe', 'RecipeDeleteSingleWithRecipeWithNoise', 'SimpleCalendarDeleteEvents', 'SimpleCalendarEventOnDateAtTime', 'SimpleCalendarEventsInNextWeek', 'TasksIncompleteTasksOnDate']

print("len to_do:", len(to_do))

num = 0

for filedir in os.listdir(filepath):
    task = filedir.split('_')[0]
    print("task = ", task)

    if task in to_do:
        num += 1
        os.remove(os.path.join(filepath, filedir))

print("num = ", num)