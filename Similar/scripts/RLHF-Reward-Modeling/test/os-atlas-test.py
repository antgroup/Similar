from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/mnt/prev_nas/virtual_agent/models/OS-Copilot/OS-Atlas-Pro-7B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "/mnt/prev_nas/virtual_agent/models/OS-Copilot/OS-Atlas-Pro-7B"
)

# Define the system prompt
sys_prompt = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]

2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
Custom Action 1: LONG_PRESS 
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>

Custom Action 2: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 3: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 4: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 5: PRESS_RECENT
    - purpose: Press the recent button to view or switch between recently used applications.
    - format: PRESS_RECENT
    - example usage: PRESS_RECENT

Custom Action 6: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 7: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 8: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
*Thoughts*: Clearly outline your reasoning process for current step.
*Actions*: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 
*Please output both actions and thoughts at the same time.*

*ATTENTION!!* You must answer specific executable actions based on the current page screenshot. 
This means that you cannot do actions that cannot be completed, such as opening an app that does not exist on the current page.

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: 
"""

# test_system_prompt = "Please pay attention to whether I have provided you with task instruction instructions. If not, please answer my questions like a typical artificial intelligence. Do I provide you with a picture? If yes, please describe the picture. If no, you just need to say 'no'."

instruction = """
Delete the following recipes from Broccoli app: Chicken Alfredo Pasta, Tomato Basil Bruschetta, Grilled Cheese with Tomato and Basil.

"""

# Define the input message
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text", "text": sys_prompt,
#             },
#             {
#                 "type": "image",
#                 "image": "./action_example_1.jpg",
#             },
#             # {"type": "text", "text": "Task instruction: to allow the user to enter their first name\nHistory: null" },
#             {"type": "text", "text": "Task instruction: to allow the user to log in\nHistory: null"},
#         ],
#     }
# ]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", "text": sys_prompt,
            },
            {
                "type": "image",
                "image": "https://pic.imgdb.cn/item/67194280d29ded1a8c384d17.jpg",
                # "image": "./action_example_1.jpg",
            },
            # {"type": "text", "text": "Task instruction: to allow the user to enter their first name\nHistory: null" },
            {"type": "text", "text": f"Task instruction: {instruction}\nHistory: null"},
        ],
    }
]


# Prepare the input for the model
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Post-process the output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)
# ['actions:\nCLICK <point>[[493,544]]</point><|im_end|>']
