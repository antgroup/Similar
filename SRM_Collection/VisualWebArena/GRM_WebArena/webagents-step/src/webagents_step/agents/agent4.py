from typing import List

import os
import shutil
import json
from time import sleep

from selenium import webdriver
from PIL import Image

from ..utils.utils import (
    get_asyn_config,
    get_fetch_config,
    send_request,
    parse_response,
    parse_fetch_result,
    get_default_config,
    ask_chatgpt_async_send,
    ask_chatgpt_async_fetch
)
import hashlib
import copy
import time


def send_asyn_request(model, msg, msg_key, temp=0.2):
    # print("send--model = ", model)
    param = get_asyn_config(model=model)

    param["queryConditions"]["messages"][0]["content"] = [{}]
    param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    param["queryConditions"]["messages"][0]["content"][0]["text"] = msg
    param["queryConditions"]["messageKey"] = msg_key
    param["queryConditions"]["temperature"] = temp
    # print(param['queryConditions'])
    response = send_request(param)
    # print("response = ", response)
    try:
        result = parse_response(response)
        return result["data"]["success"]
    except Exception as e:
        print(e)
        return False


def fetch_asyn_result(model, msg_key):
    param = get_fetch_config(model=model)
    param["queryConditions"]["messageKey"] = msg_key
    response = send_request(param)
    try:
        return parse_fetch_result(response)
    except Exception as e:
        print(e)
        return None


def ask_chatgpt(model, msg, temp='0.2'):
    param = get_default_config(model)
    param["queryConditions"]["messages"][0]["content"] = [{}]
    param["queryConditions"]["messages"][0]["content"][0]["type"] = "text"
    param["queryConditions"]["messages"][0]["content"][0]["text"] = msg
    param["queryConditions"]["temperature"] = temp

    # print("ask_chatgpt")
    try:
        ask_chatgpt_async_send(param)
    except Exception as e:
        # print("send error")
        # print(e)
        return False

    try:
        return ask_chatgpt_async_fetch(param)
    except Exception as e:
        # print("fetch error")
        # print(e)
        return None

def action_trans(action, observation):
    if ('click' in action) or ('hover' in action):
        element_id = action[action.find('[') + 1: action.rfind(']')]
        # print("element_id = ", element_id)
        content = ''
        for line in observation.splitlines():
            if element_id in line:
                # print("line = ", line)
                content = line[line.find(element_id) + len(element_id) + 1 + 1:]
                # print("content = ", content)
                break
        return action + content
    else:
        return action

# def action_content_trans(action_content, observation):
#     print("action_content = ", action_content)
#     if ('click' in action_content) or ('hover' in action_content) or ('type' in action_content):
#         content = action_content[action_content.find(']') + 1:]
#         print("content = ", content)
#
#         element_id = ''
#
#         for line in observation.splitlines():
#             if content in line:
#                 print("line = ", line)
#                 element_id = line[line.find('['):line.find(']')]
#                 print("element_id = ", element_id)
#                 break
#         return action_content[:action_content.find('[') - 1] + '[' + element_id + ']'
#     else:
#         return action_content


get_TR_C_prompt = '''
You are an expert in evaluating the performance of a Virtual Agent. The Virtual Agent is designed to help a human user complete specified tasks (such as web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile apps, etc.). Given the user’s goal, the current web observation, the current step number, the agent’s last action (if there exists) and the agent’s current action, your goal is to evaluate the quality of the current step's action of the agent. Please evaluate the quality of the current step's action of the agent based on the given dimensions.
<Goal>: {Instruction}
<Current Step Number>: {Steps}
<Observation>: {Observation}
<Last Action>: {Last Action}
<Current Action>: {Current Action}

<Requirements>
  I hope to build several dimensions to evaluate the quality of each step of Agent operation, and the result of each step operation is a screenshot of each screen:
  1) 【Task Relevance】. Its meaning is whether the operation of the Agent is related to achieving the <Goal>. Please do not be too strict in rating this indicator. Except for some obviously meaningless or obviously incorrect behaviors (such as clicking on blank spaces), it can be considered that this behavior is task related. Some steps of action may result in the task being unable to be completed but are related to the task (such as taking incorrect notes), while some steps of action may be meaningless but still result in the task being completed (such as clicking on a blank space on the screen without generating any response). This indicator can determine whether the step is related to the task goal.
  2) 【Coherence】. It represents the compactness and coherence between the current step and the previous step. If the current step is the first step, the default value of this indicator is 1. Some operations, although task related, not inefficient, and highly likely to lead to success, lack coherence with the previous step. For example, the task is to "query the Lakers' game results and record them in the Note". The Agent operations are as follows: a Open the browser; b. Open Note; c. Create new notes; d. Search for Lakers games; e. Query the results of the competition; f. Record the results of the competition in your notes. It can be found that the operations of a and b lack coherence, and it is more in line with human preferences to directly search for competition results after opening the browser instead of simultaneously opening Note.
  Please evaluate the quality of each step of the Agent operation based on the dimensions I have provided. The results of both dimensions are represented as 0 or 1. For example, if the steps have 【Task Relevance】, the score for task relevance is 1, otherwise it is 0. If the steps have 【Coherence】, then the coherence score is 1, otherwise it is 0.
</Requirements>

<Output Example>
    <System Answer>
    + **Evaluation**:
      + **Task Relevance**: 1. Reason: ...
      + **Coherence**: 0. Reason: ...
    </System Answer>
</Example>
'''


class Node:
    def __init__(self):
        self.env = None
        self.observation = None
        self.url = None
        self.step_now = 0
        self.subnode = 0

        self.action = None
        self.reason = None
        self.status = None

        self.IP = 0 # 推导潜力
        self.E = 0 # 高效性
        self.AC = 0
        self.TC = 0 # 任务贡献
        self.TR = 0 # 任务相关性
        self.C = 0 # 连贯性

        self.success = 0
        self.length = 0

    def __lt__(self, other):
        if self.success >= other.success:
            if (self.success == other.success):
                return self.TC > other.TC
        else:
            return False


class Agent:
    def __init__(
        self,
        max_actions,
        verbose=0,
        logging=False,
        previous_actions: List = None,
        previous_reasons: List = None,
        previous_responses: List = None,
    ):
        self.previous_actions = [] if previous_actions is None else previous_actions 
        self.previous_reasons = [] if previous_reasons is None else previous_reasons
        self.previous_responses = [] if previous_responses is None else previous_responses
        self.max_actions = max_actions
        self.verbose = verbose
        self.logging = logging
        self.trajectory = []
        self.data_to_log = {}

    # 重置
    def reset(self):
        self.previous_actions = []
        self.previous_reasons = []
        self.previous_responses = []
        self.trajectory = []
        self.data_to_log = {}

    # 获取历史轨迹
    def get_trajectory(self):
        return self.trajectory

    # 更新历史记录（行动 或 推理）
    def update_history(self, action, reason):
        if action:
            self.previous_actions += [action]
        if reason:
            self.previous_reasons += [reason]

    def previous_history(self):
        previous_history = []

        if len(self.previous_actions) == len(self.previous_responses):
            for action, response in zip(self.previous_actions, self.previous_responses):
                if response:
                    previous_history.append(f"{response} = {action}")
                else:
                    previous_history.append(action)
            previous_history = "\n".join(previous_history)
        else:
            previous_history = "\n".join(action for action in self.previous_actions if
                                         action is not None) if self.previous_actions is not None else ""

        return previous_history

    # 预测行为
    def predict_action(self, objective, observation, url=None):
        pass

    # 接收反馈/回复
    def receive_response(self, response):
        self.previous_responses += [response]

    def get_TR_C(self, objective, steps, observation, action_last, action):
        print("\nget_TR_C")
        now_prompt = get_TR_C_prompt.replace("{Instruction}", objective)
        now_prompt = now_prompt.replace("{Steps}", str(steps))
        now_prompt = now_prompt.replace("{Observation}", observation)
        now_prompt = now_prompt.replace("{Last Action}", action_last)
        now_prompt = now_prompt.replace("{Current Action}", action)
        # print("now_prompt = \n", now_prompt)
        # print("{Goal} = ", objective)
        # print("{Current Step Number} = ", steps)
        # print("{Last Action} = ", action_last)
        # print("{Current Action} = ", action)

        model = "gpt-4-turbo"
        temperature = 0.3
        temperature_str = str(temperature)

        msg = now_prompt
        # msg_key = hashlib.sha256(str(msg + temperature_str).encode('utf-8')).hexdigest()

        flag = 1

        while flag:

            # result = send_asyn_request(model, msg, msg_key, temperature)

            # response = fetch_asyn_result(model, msg_key)

            # sleep(0.5)
            response = ask_chatgpt(model, msg, temperature_str)
            if response != None and response != False:
                flag = 0
                # print("yes111")
                # print(response)
            else:
                flag = 1

        return response

    # 行动
    def act(self, objective, env):
        N = 3
        sum = N * (N + 1)

        # 创建两个节点数组
        arr1 = [Node() for _ in range(25)]
        arr2 = [Node() for _ in range(sum + 1)]

        # 生成的数据存储的位置
        filepath = "generate_data/9.27/" + env.config_file[:-5]
        # print("filepath = ", filepath)
        if not os.path.isdir(filepath):  # 如果没有该文件则新建一个
            os.makedirs(filepath)

        all_length_filename = filepath + '/all_length.text'

        # 初始url
        url = env.get_url()

        if not os.path.exists(all_length_filename):

            sum_length0 = 0
            num_success0 = 0

            # 从初识节点先模拟几个点，得到all_length
            for j in range(1, N + 1):
                # last_action = ""
                print("\nnode 0 sub-node %d" % j)
                env_copy = copy.copy(env) # 浅拷贝一个env
                observation_copy = env_copy.observation()
                # print("observation_copy = \n", observation_copy)
                # url_copy = env_copy.get_url()
                # print("url_copy = ", url_copy)

                print("url = ", url)
                env_copy.goto_ano(url)

                filepath0 = filepath + '-0' + '-' + str(j)

                # rollout
                while not env_copy.done():
                    observation_copy = env_copy.observation()
                    # print("observation_copy = \n", observation_copy)
                    url_copy = env_copy.get_url()
                    # 获取预测的行动和推理原因
                    action, reason = self.predict_action(objective=objective, observation=observation_copy, url=url_copy)
                    action_content = action_trans(action, observation_copy)
                    status0 = env_copy.step(action)

                    print("action_content = ", action_content)
                    # self.get_TR_C(objective, env_copy.steps, observation_copy, last_action, action_content)
                    # last_action = action

                env_copy.is_done = False # 重置是否完成的标签
                # env_copy.steps = 0

                # 如果当前模拟的节点能够完成任务
                if (status0['success']):
                    print("node 0 subnode%d success\n" % j)
                    num_success0 += 1
                    sum_length0 = sum_length0 + status0['num_actions']

            # 如果模拟的节点都不能完成任务，则跳过这个任务
            if (num_success0 == 0):
                print("pass!!!")
                return status0


            # 计算all_length
            print("sum_length0 = ", sum_length0)
            print("num_success0 = ", num_success0)
            all_length = sum_length0/num_success0
            # all_length = 10
            print("all_length = %f\n" % all_length)
            last_length = all_length

            with open(all_length_filename, 'w') as f:
                f.write(str(all_length))
        else:
            print("all_length exists!")
            with open(all_length_filename, "r", encoding='utf-8') as f:
                all_length = float(f.readline())
            print("all_length = %f\n" % all_length)
            last_length = all_length


        print("pre-action is over------------------------------------\n")
        # all_length = 0
        # last_length = 0
        avg_length = 0
        last_action = ""

        # 回到初识页面
        # env.goto(url)

        num = 1
        # arr1[1].env = env

        # screenfile = filepath + '/webpage0.png'
        # print("screenfile = ", screenfile)
        # env.screenshot(screenfile)

        status = {}

        for j in range(1, N + 1):
            print("\nsub-node %d --------------------------------------------" % (j))

            env_copy = copy.copy(env)

            # print("\nurl = ", url)
            env_copy.goto_ano(url)
            env_copy.steps = 0

            # 从下一步开始模拟
            print("\nstart rollouting-------------------------------------------")
            filepath2 = filepath + '-' + str(j)
            print("filepath2 = ", filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            while not env_copy.done():
                print("subnode%d step_now = %d" % (j, env_copy.steps + 1))

                observation_copy = env_copy.observation()
                # print("\nobservation_copy = \n", observation_copy)
                url_copy = env_copy.get_url()
                print("url_copy = ", url_copy)

                screenfile = filepath2 + '/webpage' + str(env_copy.steps + 1) + '.png'
                print("screenfile = ", screenfile)
                env_copy.screenshot(screenfile)

                # 获取预测的行动和推理原因
                action, reason = self.predict_action(
                    objective=objective, observation=observation_copy, url=url_copy
                )
                action_content = action_trans(action, observation_copy)
                print("action_content = ", action_content)

                status_now = env_copy.step(action)

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = env_copy.steps
                temp_dict['action'] = action_content  # 转换成内容
                temp_dict['reason'] = reason
                temp_dict['status'] = status_now
                # 简化指标计算
                temp_dict['IP'] = 1
                temp_dict['E'] = 1 / all_length
                temp_dict['TC'] = 1 / all_length
                temp_dict['TR'] = 1
                temp_dict['C'] = 1

                now[str(env_copy.steps)] = temp_dict

                filename_now = filepath2 + '/evaluation_score.json'

                # print("filename_now = ", filename_now)

                if (os.path.exists(filename_now)):  # 如果已经存在文件
                    with open(filename_now, 'r') as f:
                        content = json.load(f)
                    content.update(now)
                    # 更新数据
                    with open(filename_now, 'w') as f_new:
                        json.dump(content, f_new)
                else:  # 如果没有文件
                    with open(filename_now, 'w') as f:
                        json.dump(now, f)

                if (env_copy.steps > all_length + 2):
                    break

            screenfile = filepath2 + '/webpage' + str(env_copy.steps + 1) + '.png'
            print("screenfile = ", screenfile)
            env_copy.screenshot(screenfile)

            env_copy.is_done = False

            if (env_copy.steps > all_length):
                print("subnode%d step length too long!" % (j))
                if os.path.exists(filepath2):
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)
                continue

            # 如果当前节点是成功的
            if (status_now['success']):
                status = status_now

                print("subnode%d success" % (j))
                avg_length = status_now['num_actions']
                print("avg_length = ", avg_length)

                avg_length_filename = filepath2 + '/avg_length.text'
                with open(avg_length_filename, 'w') as f:
                    f.write(str(avg_length))

                # 更新E、TC的值
                with open(filename_now, 'r', encoding='UTF-8') as f:
                    dict_now = json.load(f)
                    content_now = dict_now

                for (key, value) in dict_now.items():
                    temp = {}
                    value_now = value
                    value_now['E'] = 1 / avg_length
                    value_now['TC'] = 1 / avg_length
                    temp[key] = value_now
                    content_now.update(temp)
                content_now_new = dict(sorted(content_now.items(), reverse=False))  # 重新排序
                with open(filename_now, 'w') as f_new:
                    json.dump(content_now_new, f_new)
            else:
                if os.path.exists(filepath2):
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)

            print("\n")


            # if "stop" in action:
            #     env.is_done = True
            #     break
            #
            # if step_now >= self.max_actions:
            #     break

            # if self.logging:
            #     self.log_step(
            #         objective=objective,
            #         url=env.get_url(),
            #         observation=observation,
            #         action=action,
            #         reason=reason,
            #         status=status,
            #     )

            # if len(self.previous_actions) >= self.max_actions:
            #     print(f"Agent exceeded max actions: {self.max_actions}")
            #     break

        return status # 存的是当前任务进行的状态

    # 异步行动
    async def async_act(self, objective, env):
        while not env.done():
            observation = await env.observation()
            action, reason = self.predict_action(
                objective=objective, observation=observation, url=env.get_url()
            )
            status = await env.step(action)

            if self.logging:
                self.log_step(
                    objective=objective,
                    url=env.get_url(),
                    observation=observation,
                    action=action,
                    reason=reason,
                    status=status,
                )

            if len(self.previous_actions) >= self.max_actions:
                print(f"Agent exceeded max actions: {self.max_actions}")
                break

        return status


    # 记录步骤
    def log_step(self, objective, url, observation, action, reason, status):
        self.data_to_log['objective'] = objective
        self.data_to_log['url'] = url
        self.data_to_log['observation'] = observation
        self.data_to_log['previous_actions'] = self.previous_actions[:-1]
        self.data_to_log['previous_responses'] = self.previous_responses[:-1]
        self.data_to_log['previous_reasons'] = self.previous_reasons[:-1]
        self.data_to_log['action'] = action
        self.data_to_log['reason'] = reason
        for (k, v) in status.items():
            self.data_to_log[k] = v
        self.trajectory.append(self.data_to_log)
        self.data_to_log = {}
