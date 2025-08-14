import os
import base64

from src.webagents_step.utils.utils import (
    get_default_config,
    ask_chatgpt
)

def ask_chatgpt2(model, msg, temp='0.2'):
    param = get_default_config(model)
    param["queryConditions"]["messages"] = msg
    param["queryConditions"]["temperature"] = temp
    try:
        return ask_chatgpt(param)
    except Exception as e:
        # print("error")
        # print(e)
        return False

img_file = "generate_data/tasks/vwa/test_shopping/1-0-1/webpage1.png"
with open(img_file, 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant performing tasks on a web browser. To solve these tasks, you will issue specific actions."
            }
        ],
    }
]

messages.append(
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "IMAGES: (1) current page screenshot is showed below, please describe the content in the image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    # "url": pil_to_b64(img),
                    # "url": f'data:image/png;base64,{encode_image(img)}'
                    "url": f'data:image/png;base64,{base64_image}'
                },
            },
        ]
    }
)

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "May I ask you what question I just asked you?"
            }
        ],
    }
]


model = 'gpt-4o'

msg_list = [messages, messages2]


for msg in msg_list:
    while True:
        response = ask_chatgpt2(model, msg)
        if (response != None) and (response != False):
            flag = 0
            print("\nyes\n")
            print(
                "\nresponse = ----------------------------------------------------------------------------------------\n",
                response)
            print(
                "---------------------------------------------------------------------------------------------------\n")
            break
        else:
            flag = 0
            # print("\nno\n")