from typing import List

import os
import shutil
import json
from time import sleep
import numpy as np
import base64
import io

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

import requests

upload_image_url = "https://api.superbed.cn/upload"


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format='PNG')
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes

def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)

def encode_image(image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')


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
        print("send error")
        print(e)
        return False

    try:
        return ask_chatgpt_async_fetch(param)
    except Exception as e:
        print("fetch error")
        print(e)
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



def get_similar_score(instruction, reason_steps, observation_url, action):
    # 设置请求头
    headers = {'Content-Type': 'application/json'}

    # 直接调用生产环境会报错
    # url = "http://cv.gateway.alipay.com/ua/invoke"
    url = "http://cv-cross.gateway.alipay.com/ua/invoke"

    data = {
        "serviceCode": "Reward_Model_ArmoRM",
        "uri": "ArmoRM_Llama3_8B_v02",
        "attributes": {
            "_TIMEOUT_": "300000",
            "_ROUTE_": "MAYA",
            # "_APP_TOKEN_": "your-app-token"
        },
        "params": {
            "features": {
                "instruction": instruction,
                "reason_steps": reason_steps,
                "observation_url": observation_url,
                "action": action,
            }
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    print("response.text = ", response.text)

    while not 'resultMap' in response.text:
        response = requests.post(url, headers=headers, data=json.dumps(data))

    data = json.loads(response.text)

    # print("get_similar_score data:", data)
    score_map = data['resultMap']['objectAttributes']
    return score_map



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
        return self.IP + self.E + self.TC + self.TR + self.C > other.IP + other.E + other.TC + other.TR + other.C


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
    def predict_action(self, objective, observation, img, url=None):
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

        model = "gpt-4o"
        temperature = 0.3
        temperature_str = str(temperature)

        msg = now_prompt
        # msg_key = hashlib.sha256(str(msg + temperature_str).encode('utf-8')).hexdigest()

        flag = 1

        while flag:
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
        arr2 = [Node() for _ in range(sum + 1)]
        action_list = []

        # 生成的数据存储的位置
        filepath = "generate_data/tasks/vwa/test_classifieds/" + env.config_file[len("tasks/vwa/test_classifieds/") : -5]
        print("filepath = ", filepath)
        if not os.path.isdir(filepath):  # 如果没有该文件则新建一个
            os.makedirs(filepath)

        action_list_file = filepath + '/action_list.json'
        if not os.path.exists(action_list_file):
            now = {}
            with open(action_list_file, 'w') as f:
                json.dump(now, f)



        step_exist = 0

        action_list_file = filepath + '/action_list.json'
        if not os.path.exists(action_list_file):
            now = {}
            with open(action_list_file, 'w') as f:
                json.dump(now, f)
        else:
            with open(action_list_file, 'r', encoding='UTF-8') as f:
                dict_now = json.load(f)

            for (key, value) in dict_now.items():
                action_list.append(value['action'])
                last_action = value['action_content']
                step_exist += 1

            print("action_list = ", action_list)

            filepath_past = filepath + '/webpage' + str(step_exist + 1) + '.png'
            if os.path.exists(filepath_past):
                print("remove filepath_past = ", filepath_past)
                os.remove(filepath_past)

            for j in range(1, N + 1):
                filepath_past = filepath + '-' + str(step_exist + 1) + '-' + str(j)
                if os.path.exists(filepath_past):
                    print("remove filepath_past = ", filepath_past)
                    shutil.rmtree(filepath_past)



        # 回到初识页面
        env_copy = copy.copy(env)
        env_copy.reset()
        # env.goto(url)


        status = {}
        env.steps = step_exist

        while not env.done():
            suc = [0 for _ in range(N + 1)]
            # 当前步骤序号
            step_now = env.steps
            print("step_now = %d ---------------------------------------------------------------" % (step_now + 1))  # 当前要执行的步骤序号

            # 当前页面
            observation = env.observation()
            # print("observation = ", observation)
            url = env.get_url()
            # print("url = ", url)


            for j in range(1, N + 1):
                print("\nsub-node %d --------------------------------------------" % (j))

                env_copy = copy.copy(env)

                # print("\nurl = ", url)
                env_copy.reset()
                # env_copy.goto_ano(url_ori)
                # observation_copy = env_copy.observation()
                # print("observation_copy = ", observation_copy)


                filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
                print("filepath2 = ", filepath2)
                if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                    os.makedirs(filepath2)

                instruction = objective
                trajectory = "Let\'s complete the task step by step. \n\n"

                action_id = 0

                # 执行到第n步
                env_copy.steps = 0
                print("\nrecall action_list\n")
                for action in action_list:
                    action_id += 1
                    observation_copy = env_copy.observation()
                    print("observation_copy = \n", observation_copy)
                    print("action_now = ", action)
                    action_content = action_trans(action, observation_copy)
                    print("action_content = \n", action_content)
                    env_copy.step(action)
                    trajectory += "Step " + str(action_id) + " : " + action + "\n\n"

                env_copy.steps = step_now

                observation_copy = env_copy.observation()
                url_copy = env_copy.get_url()

                screenfile = filepath + '/webpage' + str(step_now + 1) + '.png'
                print("screenfile = ", screenfile)
                if not os.path.exists(screenfile):
                    env_copy.screenshot(screenfile)

                screenfile_sub = filepath2 + '/webpage' + str(step_now + 1) + '.png'  # 主轨迹的页面截图
                env_copy.screenshot(screenfile_sub)

                # screen_url_file = filepath + '/webpage' + str(step_now + 1) + 'url.txt'  # 主轨迹的页面截图
                # observation_url = ''
                # if not os.path.exists(screen_url_file):
                #     resp = requests.post(upload_image_url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"},
                #                          files={"file": open(screenfile, "rb")})
                #     observation_url = resp.json()['url']
                # else:
                #     with open(screen_url_file, "r", encoding='utf-8') as f:
                #         observation_url = f.readline()
                # print("observation_url = ", observation_url)
                observation_url = ''


                print("\nDo right now step!")

                with open(screenfile, 'rb') as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                # 执行当前步骤的操作
                action, reason = self.predict_action(objective=objective, observation=observation_copy, img=base64_image, url=url_copy)
                action_content = action_trans(action, observation_copy)
                print("observation_copy = \n", observation_copy)
                print("action_now = ", action)
                print("action_content = ", action_content)

                status = env_copy.step(action)


                # 存储的是执行动作后的状态
                arr2[j].env = env_copy
                arr2[j].url = url_copy
                arr2[j].observation = observation_copy
                arr2[j].action = action
                arr2[j].reason = reason
                arr2[j].status = status
                arr2[j].subnode = j

                status_now = status


                score_map = get_similar_score(instruction, trajectory, observation_url, action_content)
                print("\nscore_map = ", score_map)
                print("\n")

                arr2[j].IP = score_map["out_IP"]
                arr2[j].E = score_map["out_E"]
                arr2[j].TC = score_map["out_TC"]
                arr2[j].TR = score_map["out_TR"]
                arr2[j].C = score_map["out_C"]


                filename_now = filepath2 + '/evaluation_score.json'
                filename = filepath + '/evaluation_score.json'
                if not os.path.exists(filename_now):
                    now = {}
                    with open(filename_now, 'w') as f:
                        json.dump(now, f)
                    if os.path.exists(filename):
                        shutil.copyfile(filename, filename_now)

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = step_now + 1
                temp_dict['action'] = action_content
                temp_dict['reason'] = reason

                temp_dict['IP'] = arr2[j].IP
                temp_dict['E'] = arr2[j].E
                temp_dict['TC'] = arr2[j].TC
                temp_dict['TR'] = arr2[j].TR
                temp_dict['C'] = arr2[j].C

                now[str(step_now + 1)] = temp_dict

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

                env_copy.is_done = False


            # 对arr2进行排序
            sorted(arr2)

            # 确定step_TC的值
            step_IP = arr2[1].IP
            step_E = arr2[1].E
            step_TC = arr2[1].TC
            step_TR = arr2[1].TR
            step_C = arr2[1].C

            # 更新当前的env环境
            # env = arr2[1].env
            env.steps = step_now + 1
            observation_copy = arr2[1].observation
            # url = env.get_url()
            # print("final url = ", url)

            # 确定当前步骤的action、reason
            action = arr2[1].action
            reason = arr2[1].reason
            status = arr2[1].status
            # print("actual step action")
            # print("actual action = ", action)
            action_content = action_trans(action, observation_copy) # 转换之后的action
            print("actual action_content = ", action_content)

            # 更新action_list
            action_list.append(action)
            action_now = {}
            temp_dict = {}
            temp_dict['action'] = action
            temp_dict['action_content'] = action_content
            temp_dict['reason'] = reason
            action_now[step_now + 1] = temp_dict
            if (os.path.exists(action_list_file)):  # 如果已经存在文件
                with open(action_list_file, 'r') as f:
                    content = json.load(f)
                content.update(action_now)
                # 更新数据
                with open(action_list_file, 'w') as f_new:
                    json.dump(content, f_new)


            # 总结并存储当前步骤的数据
            print("\nstep_now : ", step_now + 1)
            print("action = ", action)
            print("action_content = ", action_content)
            print("step_IP = %f" % step_IP)
            print("step_E = %f" % step_E)
            print("step_TC = %f" % step_TC)
            print("step_TR = %f" % step_TR)
            print("step_C = %f\n" % step_C)

            now = {}
            temp_dict = {}
            temp_dict['step_idx'] = step_now + 1
            temp_dict['action'] = action_content  # 转换成内容
            temp_dict['reason'] = reason
            temp_dict['status'] = status
            temp_dict['IP'] = step_IP
            temp_dict['E'] = step_E
            temp_dict['TC'] = step_TC
            temp_dict['TR'] = step_TR
            temp_dict['C'] = step_C

            now[str(step_now + 1)] = temp_dict

            filename = filepath + '/evaluation_score.json'
            # print("filename = ", filename)

            if (os.path.exists(filename)):  # 如果已经存在文件
                with open(filename, 'r') as f:
                    content = json.load(f)
                content.update(now)
                # 更新数据
                with open(filename, 'w') as f_new:
                    json.dump(content, f_new)
            else:  # 如果没有文件
                with open(filename, 'w') as f:
                    json.dump(now, f)



            if "stop" in action:
                env.is_done = True
                break

            if step_now >= self.max_actions:
                # env.is_done = True
                print(f"Agent exceeded max actions: {self.max_actions}")
                break

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

        status_file = filepath + '/status.json'
        if not os.path.exists(status_file):
            with open(status_file, 'w') as f:
                json.dump(status, f)

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
