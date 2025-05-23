from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

# Create Reward Model Pipeline
rm = ArmoRMPipeline("/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/similar/similar_llava", trust_remote_code=True)

# score the messages
# prompt = 'What are some synonyms for the word "beautiful"?'
# response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'

# score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])

messages1 = [{'role': 'system', 'content': [{'type': 'text', 'text': '\nYou are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks \n(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile \ndevices, operation systems, etc.) based on given instructions.\n\nYou will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:\n1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.\n2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).\n3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from \nwhich you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.\n'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': "\n[INSTRUCTION]: In Simple Calendar Pro, create a recurring calendar event titled 'Review session for Budget Planning' starting on 2023-10-15 at 14h. The event recurs weekly, forever, and lasts for 60 minutes each occurrence. The event description should be 'We will understand business objectives. Remember to confirm attendance.'.\n[OBSERVATION]: which is a single image provided.\n[REASON_STEPS]: Step 1 :  open_app Simple Calendar Pro\n\nStep 2 : click New Event\n\nStep 3 : click Event\n\nStep 4 :  input_text Review session for Budget Planning\n\n\n"}, {'type': 'image', 'image': 'https://pic.imgdb.cn/item/6719965bd29ded1a8c609b32.jpg'}]}, {'role': 'assistant', 'content': 'click 16:00'}]

score1 = rm(messages1)

print("score1 :", score1)

messages2 = [{'role': 'system', 'content': [{'type': 'text', 'text': '\nYou are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks \n(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile \ndevices, operation systems, etc.) based on given instructions.\n\nYou will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:\n1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.\n2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).\n3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from \nwhich you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.\n'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': "\n[INSTRUCTION]: In Simple Calendar Pro, create a recurring calendar event titled 'Review session for Budget Planning' starting on 2023-10-15 at 14h. The event recurs weekly, forever, and lasts for 60 minutes each occurrence. The event description should be 'We will understand business objectives. Remember to confirm attendance.'.\n[OBSERVATION]: which is a single image provided.\n[REASON_STEPS]: Step 1 :  open_app Simple Calendar Pro\n\nStep 2 : click New Event\n\nStep 3 : click Event\n\nStep 4 :  input_text Review session for Budget Planning\n\n\n"}, {'type': 'image', 'image': 'https://pic.imgdb.cn/item/6719965bd29ded1a8c609b32.jpg'}]}, {'role': 'assistant', 'content': ' input_text We will understand business objectives. Remember to confirm attendance.'}]

score2 = rm(messages2)

print("score2 :", score2)
