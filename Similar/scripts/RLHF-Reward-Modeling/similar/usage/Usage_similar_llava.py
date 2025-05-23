import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda"
path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/similar/similar_llava"
model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

# We load a random sample from the validation set of the HelpSteer dataset  我们从HelpSteer数据集的验证集中加载随机样本
prompt = 'What are some synonyms for the word "beautiful"?'
response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
# messages = [{"role": "user", "content": prompt},
#            {"role": "assistant", "content": response}]

messages = [{'role': 'system', 'content': [{'type': 'text', 'text': '\nYou are a virtual agent. The Virtual Agent is designed to help a human user complete specified tasks \n(such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile \ndevices, operation systems, etc.) based on given instructions.\n\nYou will predict the next action based on following content [INSTRUCTION], [OBSERVATION], [REASON_STEPS]:\n1. [INSTRUCTION]: It is your ultimate goal, and all your actions are aimed at completing this task.\n2. [OBSERVATION]: It is an observation of an image, which is the screenshot of the platform (such as computer screen).\n3. [REASON_STEPS]: They are the trajectory of the actions you performed in the past to complete the instruction, from \nwhich you can understand how you thought in order to complete the instruction. If it is empty, it means it is currently the first step.\n'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '\n[INSTRUCTION]: Add the following expenses into the pro expense:\nExpense: Therapy Sessions\n amount_dollars: $307.01\n category_name: Health Care\n note: I may repeat this\n\n[OBSERVATION]: which is a single image provided.\n[REASON_STEPS]: \n'}, {'type': 'image', 'image': 'https://pic.imgdb.cn/item/6719427ad29ded1a8c3847ce.jpg'}]}, {'role': 'assistant', 'content': 'click Search'}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

with torch.no_grad():
   output = model(input_ids)
   # Multi-objective rewards for the response  响应的多目标奖励
   multi_obj_rewards = output.rewards.cpu().float()
   # The gating layer's output is conditioned on the prompt  门控层的输出取决于提示
   gating_output = output.gating_output.cpu().float()
   # The preference score for the response, aggregated from the multi-objective rewards with the gating layer  响应的偏好得分，由门控层的多目标奖励聚合而成
   preference_score = output.score.cpu().float()

print("multi_obj_rewards = ", multi_obj_rewards)
print("gating_output = ", gating_output)
print("preference_score = ", preference_score)

# We apply a transformation matrix to the multi-objective rewards before multiplying with the gating layer's output. This mainly aims at reducing the verbosity bias of the original reward objectives
# 在与门控层的输出相乘之前，我们将变换矩阵应用于多目标奖励。这主要是为了减少原始奖励目标的冗长偏见
obj_transform = model.reward_transform_matrix.data.cpu().float()
# The final coefficients assigned to each reward objective 分配给每个奖励目标的最终系数
multi_obj_coeffs = gating_output @ obj_transform.T

score = torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1)
print("score = ", score)
# The preference score is the linear combination of the multi-objective rewards with the multi-objective coefficients, which can be verified by the following assertion
# 偏好得分是多目标奖励与多目标系数的线性组合，可以通过以下断言进行验证
assert torch.isclose(torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1), preference_score, atol=1e-1)

# Find the top-K reward objectives with coefficients of the highest magnitude  找到系数最高的前K个奖励目标
K = 5
top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True,)[:, :K]
top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

# The attributes of the 5 reward objectives  5个奖励目标的属性
attributes = [
    "IP",
    "E",
    "TC",
    "TR",
    "C",
]


example_index = 0
for i in range(K):
   attribute = attributes[top_obj_dims[example_index, i].item()]
   coeff = top_obj_coeffs[example_index, i].item()
   print(f"{attribute}: {round(coeff, 5)}")
# code-complexity: 0.19922
# helpsteer-verbosity: -0.10864
# ultrafeedback-instruction_following: 0.07861


# The actual rewards of this example from the HelpSteer dataset are [3,3,4,2,2] for the five helpsteer objectives:
# helpfulness, correctness, coherence, complexity, verbosity
# We can linearly transform our predicted rewards to the original reward space to compare with the ground truth
# helpsteer_rewards_pred = multi_obj_rewards[0, :5] * 5 - 0.5
# print(helpsteer_rewards_pred)
# [2.78125   2.859375  3.484375  1.3847656 1.296875 ]