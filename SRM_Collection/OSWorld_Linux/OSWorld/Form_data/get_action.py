
# data = {"action": ["import pyautogui\nimport time\n\n# Click on the \"Show Applications\" button\npyautogui.click(35, 1045)  # Coordinates for the \"Show Applications\" button\ntime.sleep(1)"]}
# data = {"action": ["import pyautogui\nimport time\n\n# Click to select slide 5, ensuring it's in focus\npyautogui.click(100, 550)\ntime.sleep(1)\n\n# Simulate selecting all textboxes (Ctrl+A might not work well for textboxes specifically)\npyautogui.hotkey('ctrl', 'a')\ntime.sleep(0.5)\n\n# Open the font color dropdown - position adjusted for yellow\npyautogui.click(300, 100)  # Adjust this if needed for the font color button\ntime.sleep(0.5)\n\n# Select yellow color\npyautogui.click(350, 200)  # Adjust to the position of yellow in the font color palette\ntime.sleep(1)"]}
data = {"action": []}

# for action in data["action"]:
#     now_action_list = action.split("\n")
#     # print(now_action_list)
#     for now_action in now_action_list:
#         # print(len(now_action))
#         if(len(now_action)<1):
#             continue
#         if (now_action[0] == '#'):
#             print(now_action)
#     print("\n")

def get_action(action_list):
    if len(action_list) == 0:
        return 'hover'

    res = ''
    for action in action_list:
        now_action_list = action.split("\n")
        # print(now_action_list)
        for now_action in now_action_list:
            # print(len(now_action))
            if (len(now_action) < 1):
                continue
            if (now_action[0] == '#'):
                res += now_action[2:] + ' \n'
                # print(now_action)
        # print("\n")
    return res

print("action = ", get_action(data['action']))
