from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


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
            multi_obj_rewards = output.rewards.cpu().float()
            gating_output = output.gating_output.cpu().float()
        return {"score": score, "multi_obj_rewards": multi_obj_rewards, 'gating_output': gating_output}

# Create Reward Model Pipeline
rm = ArmoRMPipeline("/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/similar/similar_llava", trust_remote_code=True)

csv = pd.read_csv("/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset_test_modify.csv")

num_all = 0
num_sum = 0
for index, row in csv.iterrows():
    type_now = row['type']
    num_all += 1
    messages_chosen = eval(row['chosen'])
    messages_rejected = eval(row['rejected'])

    output_chosen = rm(messages_chosen)
    output_rejected = rm(messages_rejected)

    score_chosen = output_chosen['score']
    score_rejected = output_rejected['score']

    multi_obj_rewards_chosen = output_chosen['multi_obj_rewards'].tolist()
    multi_obj_rewards_rejected = output_rejected['multi_obj_rewards'].tolist()

    # print("multi_obj_rewards_chosen = ", multi_obj_rewards_chosen)

    IP_chosen = multi_obj_rewards_chosen[0][0]
    E_chosen = multi_obj_rewards_chosen[0][1]
    TC_chosen = multi_obj_rewards_chosen[0][2]
    TR_chosen = multi_obj_rewards_chosen[0][3]
    C_chosen = multi_obj_rewards_chosen[0][4]
    total_chosen = IP_chosen * 5 + E_chosen * 5 + TC_chosen * 3 + TR_chosen + C_chosen

    IP_rejected = multi_obj_rewards_rejected[0][0]
    E_rejected = multi_obj_rewards_rejected[0][1]
    TC_rejected = multi_obj_rewards_rejected[0][2]
    TR_rejected = multi_obj_rewards_rejected[0][3]
    C_rejected = multi_obj_rewards_rejected[0][4]
    total_rejected = IP_rejected * 5 + E_rejected * 5 + TC_rejected * 3 + TR_rejected + C_rejected

    if (type_now == 'IP') and (IP_chosen > IP_rejected):
        num_sum += 1
    elif (type_now == 'E') and (E_chosen > E_rejected):
        num_sum += 1
    elif (type_now == 'TC') and (TC_chosen > TC_rejected):
        num_sum += 1
    elif (type_now == 'TR') and (TR_chosen > TR_rejected):
        num_sum += 1
    elif (type_now == 'C') and (C_chosen > C_rejected):
        num_sum += 1
    elif (type_now == 'total') and (total_chosen > total_rejected):
        num_sum += 1

    # if score_chosen > score_rejected:
    #     num_sum += 1

    if num_all % 100 == 0:
        print("accuracy = ", num_sum / num_all)

print("accuracy total = ", num_sum / num_all)
