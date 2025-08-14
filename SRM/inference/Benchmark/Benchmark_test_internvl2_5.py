# 多线程并发版
# 为每张图像的观感打分
import pandas as pd
import requests
import os
from tqdm import tqdm
import json
import concurrent.futures
from functools import partial
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import requests as req

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

system_prompt = '''
  You are an expert in evaluating the performance of a Virtual Agent. The Virtual Agent is designed to help a human user complete specified tasks (such as app usage, web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile devices, operation systems, etc.) based on given instructions. 
  Given the user’s *INSTRUCTION*, the *OBSERVATION* of current platforms, the action *TRAJECTORY* of the agent, the two *ACTION_1* and *ACTION_2* predicted by the agent, and the current action step number *STEP_IDX*. Your GOAL is to help me complete *step-wise evaluation*, that is, *evaluate the quality of the Agent's ACTION* in a specific dimension. 
  Please evaluate the quality of the Agent ACTION based one of the given <EVALUATION DIMENSION> and determine which one is better: *ACTION_1* or *ACTION_2*. 
  If *ACTION_1* is better, please output "1" and the Reason; If *ACTION_2* is better, please output "2" and the Reason. 
  Note that please do not output answers such as 'two actions are similar'. Please make a choice and output the number and corresponding reason.
\n

<Word Meaning>
1. *INSTRUCTION*: refers to the command of human users to the Agent, which is the specific content that the Agent needs to complete the task on a specific platform, that is, the ultimate GOAL of the Agent.
2. *OBSERVATION*: refers to the specific information of the current platform state that an agent can observe on the platform where the task needs to be completed, which is the environment in which the agent is currently located. In our task, observations are presented in the form of images, known as screenshots.
3. *TRAJECTORY*: refers to the action prediction made by an agent in the past to complete the INSTRUCTION, which records all actions taken by the agent from the first step to the current step. If this is the first step, then the trajectory is empty.
4. *ACTION*: refers to the predicted operation of the Agent in the current state to complete the INSTARCTION in the current step. This operation generally refers to a simple action command, such as "CLICK", "TYPE", etc. Note that ACTION is the result predicted by the agent after observing the current OBSERVATION, and the Agent often cannot complete the task in one step.
5. *STEP_IDX*: refers to the sequence number of the Agent executing the current ACTION to complete the INSTRUCTION.
</Word Meaning>
\n

<EVALUATION DIMENSION>
Please evaluate the quality of each step of the Agent operation based on the dimensions provided follow, with scores given in specific dimension. The higher the value, the more satisfied this attribute is.
1. [Inference Potential]
    1.1 Meaning: It indicates the potential of the step to complete the task, which is the probability of a step reaching the completion of the task.
2. [Efficiency]
    2.1 Meaning: It indicates whether this step is efficient in completing the task. We calculate this metric as the difference between 'the number of steps required to complete the final task after the current step' and 'the number of steps required to complete the final task after the previous step', divided by 'the total number of steps required to complete the task'. This indicates the degree of efficiency improvement in completing tasks after the current step is executed.
3. [Task Contribution]
    3.1 Meaning: It indicates the degree to which this step contributes to the completion of the final task. There are good and bad contributions, the correct steps will give a positive contribution, and the wrong steps will give a negative contribution.
4. [Task Relevance]
    4.1 Meaning: It indicates is whether the operation of the Agent is related to achieving the INSTRUCTION. 
    4.2 Range of values after mapping: *{0, 1}*. The larger the value, the greater the correlation between the step and the task, that is, 1 indicates that the operation has a high task correlation, while 0 indicates that the operation has almost no correlation with the task.
5. [Coherence]. 
    5.1 Meaning: It represents the compactness and coherence between the current step and the previous step. 
    5.2 Range of values after mapping: *{0, 1}*. The larger the value, the greater the coherence of the step, that is, 1 indicates that the operation has greater coherence, which is in line with human preferences, while 0 indicates lower coherence of the operation.
6. [Total].
    Meaning: Integrated decision-making based on the 5 dimensions mentioned earlier
</EVALUATION DIMENSION>
\n

Please only return JSON format content to ensure it can be parsed by json.loads()
<Output Format>
{"choose":<1 or 2>, "reason": <Reason for judgment>}
</Output Format>
\n
'''

user_prompt = """
[EVALUATION DIMENSION]: <EVALUATION DIMENSION>
[INSTRUCTION]: <INSTRUCTION>
[OBSERVATION]: which is a single image provided.
[TRAJECTORY]: <TRAJECTORY>
[ACTION_1]: <ACTION_1>
[ACTION_2]: <ACTION_2>
[STEP_IDX]: <STEP_IDX>
"""

MAX_RETRIES = 10
N = 10


def trans_type(type_now):
    if type_now == 'IP':
        type_now = "Inference Potential"
    elif type_now == 'E':
        type_now = "Efficiency"
    elif type_now == 'TC':
        type_now = "Task Contribution"
    elif type_now == 'TR':
        type_now = "Task Relevance"
    elif type_now == 'C':
        type_now = "Coherence"
    elif type_now == 'total':
        type_now = "Total"

    return type_now


def process_single_image(row, results_list):
    # choose = row['choose']
    # if not choose == None:
    #     print("exist\n")
    #     return

    # print("row = ", row)
    compare_id = row['compare_id']
    type_now = row['type']
    type_now = trans_type(type_now)
    instruction = row["instruction"]
    observation = row["image_url"]
    trajectory = row["reason_steps"]
    if pd.isna(trajectory):
        trajectory = "Let's complete the task step by step."
    action1 = row["chosen_action"]
    action2 = row["rejected_action"]
    step_idx = row["step_idx"]

    observation_trans = req.get(observation)
    observation = observation_trans.url
    if observation.startswith("https"):
        observation = observation.replace("https", "http")

    user_message = user_prompt
    user_message = user_message.replace("<EVALUATION DIMENSION>", type_now)
    user_message = user_message.replace("<INSTRUCTION>", instruction)
    user_message = user_message.replace("<TRAJECTORY>", trajectory)
    user_message = user_message.replace("<ACTION_1>", action1)
    user_message = user_message.replace("<ACTION_2>", action2)
    user_message = user_message.replace("<STEP_IDX>", str(step_idx))

    # print("type_now = ", type_now)

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": observation},
                },
            ]
        }
    ]

    # print("compare_id = ", compare_id)
    retry_cnt = 0
    while retry_cnt < MAX_RETRIES:
        retry_cnt += 1
        try:
            response = client.chat.completions.create(
                model="InternVL2_5-8B",
                # model="Qwen2-VL-72B-Instruct",
                messages=messages,
                temperature=0.2,
            )
            response = response.choices[0].message.content
            # print(response)
            if response[0] == '`':
                response = response[8:-4]
            response = json.loads(response)
            print(response)
            choose = response['choose']
            reason = response["reason"]
            results_list.append({"compare_id": compare_id, "choose": choose, "reason": reason})
            return
        except Exception as e:
            print("error:", e)

    results_list.append({"compare_id": compare_id, "choose": None, "reason": None})


df = pd.read_csv(
    "/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset_test_modify.csv")

# rest = []
# for index, row in df.iterrows():
#     if not (row["choose"]==1.0 or row["choose"]==2.0):
#         rest.append(row["compare_id"])
# # print("rest = ", rest)

# df_rest = df.loc[df['compare_id'].isin(rest)]
# print("len(df) = ", df_rest.shape[0])

# 存储所有结果的列表
results_list = []

# 使用线程池进行并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = []
    for _, row in df.iterrows():
        future = executor.submit(process_single_image, row, results_list)
        futures.append(future)

    # 使用tqdm显示进度
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        # print("results_list = ", results_list)
        if len(results_list) % 10 == 0:
            temp_results = results_list.copy()
            for i in temp_results:
                df.loc[df['compare_id'] == i['compare_id'], 'choose'] = i['choose']
                df.loc[df['compare_id'] == i['compare_id'], 'reason'] = i['reason']
            df.to_csv('Benchmark_test_internvl2_5_1.7.csv', index=False, encoding='utf-8')

for i in results_list:
    df.loc[df['compare_id'] == i['compare_id'], 'choose'] = i['choose']
    df.loc[df['compare_id'] == i['compare_id'], 'reason'] = i['reason']
df.to_csv('Benchmark_test_internvl2_5_1.7.csv', index=False, encoding='utf-8')