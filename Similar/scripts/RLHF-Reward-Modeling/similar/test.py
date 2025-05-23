import json

import torch
import pandas as pd
from transformers import AutoTokenizer, pipeline

rm_tokenizer = AutoTokenizer.from_pretrained("/mnt/prev_nas/virtual_agent/MBC/ArmoRM/FsfairX-LLaMA3-RM-v0.1")
device = 0 # accelerator.device
rm_pipe = pipeline(
    "sentiment-analysis",
    model="/mnt/prev_nas/virtual_agent/MBC/ArmoRM/FsfairX-LLaMA3-RM-v0.1",
    #device="auto",
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1
}

# chat = [
# {"role": "user", "content": "Hello, how are you?"},
# {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
# {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]

# chat = [
#     {
#         'role': 'system',
#         'content': [
#             {
#                 'type': 'text', 'text': '\nYou are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks \n(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile \ndevices, operation systems, etc.) based on given instructions.\n\nYou will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:\n1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.\n2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).\n3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from \nwhich you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.\n'
#             }
#         ]
#     },
#     {
#         'role': 'user',
#         'content': [
#             {
#                 'type': 'text', 'text': '\n[INSTRUCTION]: Add the following expenses into the pro expense:\nExpense: Therapy Sessions\n amount_dollars: $307.01\n category_name: Health Care\n note: I may repeat this\n\n[OBSERVATION]: which is a single image provided.\n[REASON_STEPS]: \n'
#             },
#             {
#                 'type': 'image', 'image': 'https://pic.imgdb.cn/item/6719427ad29ded1a8c3847ce.jpg'
#             }
#         ]
#     },
#     {
#         'role': 'assistant',
#         'content': 'click Search'
#     }
# ]

chat1 = [{'role': 'system', 'content': [{'type': 'text', 'text': '\nYou are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks \n(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile \ndevices, operation systems, etc.) based on given instructions.\n\nYou will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:\n1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.\n2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).\n3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from \nwhich you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.\n'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '\n[INSTRUCTION]: Add the following expenses into the pro expense:\nExpense: Therapy Sessions\n amount_dollars: $307.01\n category_name: Health Care\n note: I may repeat this\n\n[OBSERVATION]: which is a single image provided.\n[REASON_STEPS]: \n'}, {'type': 'image', 'image': 'https://pic.imgdb.cn/item/6719427ad29ded1a8c3847ce.jpg'}]}, {'role': 'assistant', 'content': 'click Search'}]

dataset = pd.read_csv("/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_multi-objective_dataset.csv")
for index, row in dataset.iterrows():
    print("index = ", index)
    print("row[messages] = ", row["messages"])
    message = eval(row["messages"])
    test_texts = [rm_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]

test_texts = [rm_tokenizer.apply_chat_template(chat1, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")]
pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
rewards = [output[0]["score"] for output in pipe_outputs]
# print("rewards: ", rewards)
