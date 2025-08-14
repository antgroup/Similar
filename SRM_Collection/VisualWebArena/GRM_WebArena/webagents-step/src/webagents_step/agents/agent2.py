from typing import List

import os
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
        element_id = action[action.find('[') + 1: action.rfind(']') - 1]
        # print("element_id = ", element_id)
        content = ''
        for line in observation.splitlines():
            if element_id in line:
                content = line[line.find(element_id) + len(element_id) + 1 + 1:]
                # print("content = ", content)
                break
        return action + content
    else:
        return action


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
        self.observation = None
        self.url = None
        self.env = None
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
        print("{Goal} = ", objective)
        print("{Current Step Number} = ", steps)
        print("{Last Action} = ", action_last)
        print("{Current Action} = ", action)

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
                print(response)
            else:
                flag = 1

        return response

    # 行动
    def act(self, objective, env):
        N = 3
        sum = N * (N + 1)
        arr1 = [Node() for _ in range(N + 1)]
        arr2 = [Node() for _ in range(sum + 1)]
        # print("111111111")

        filepath = "generate_data/" + env.config_file[:-5]
        # print("filepath = ", filepath)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        url = env.get_url()

        # sum_length0 = 0
        # num_success0 = 0

        # for j in range(1, N + 1):
        #     # last_action = ""
        #     print("\nnode 0 sub-node %d" % j)
        #     # env_copy = env.clone()
        #     env_copy = copy.copy(env)
        #     observation_copy = env_copy.observation()
        #     # print("observation_copy = \n", observation_copy)
        #     # url_copy = env_copy.get_url()
        #     # print("url_copy = ", url_copy)
        #     # print("goto")
        #     print("url = ", url)
        #     env_copy.goto(url)
        #     # print("env_copy.steps = ", env_copy.steps)
        #
        #     while not env_copy.done():
        #         observation_copy = env_copy.observation()
        #         # print("observation_copy = ", observation_copy)
        #         url_copy = env_copy.get_url()
        #         # 获取预测的行动和推理原因
        #         action, reason = self.predict_action(
        #             objective=objective, observation=observation_copy, url=url_copy
        #         )
        #         # action_content = action_trans(action, observation_copy)
        #         status0 = env_copy.step(action)
        #
        #         # self.get_TR_C(objective, env_copy.steps, observation_copy, last_action, action_content)
        #         # last_action = action
        #
        #     env_copy.is_done = False
        #     # env_copy.steps = 0
        #
        #     if (status0['success']):
        #         print("node 0 subnode%d success\n" % j)
        #         num_success0 += 1
        #         sum_length0 = sum_length0 + status0['num_actions']
        #
        # if (num_success0 == 0):
        #     print("pass!!")
        #     return status0
        #
        # print("sum_length0 = ", sum_length0)
        # print("num_success0 = ", num_success0)
        # all_length = sum_length0/num_success0
        all_length = 6
        print("all_length = %f\n" % all_length)
        last_length = all_length

        print("preaction is over------------------------------------\n")
        # all_length = 0
        # last_length = 0
        avg_length = 0

        last_action = ""
        status = {}

        env.goto(url)
        while not env.done():
            observation = env.observation()
            url = env.get_url()
            # print("url1111 = ", url)
            step_now = env.steps
            print("step_now = ", step_now + 1) # 当前要执行的步骤序号
            if(step_now + 1 > all_length):
                print("node step length too long!")
                return status

            num_success = 0
            max_length = 0
            min_length = float('inf')
            sum_length = 0

            # option = webdriver.ChromeOptions()
            # option.add_argument('headless')
            # driver = webdriver.Chrome(options=option)
            # driver.get(url)
            # width = driver.execute_script("return document.documentElement.scrollWidth")
            # height = driver.execute_script("return document.documentElement.scrollHeight")
            # driver.set_window_size(1920, 1080)  # 修改浏览器窗口大小
            # driver.get_screenshot_as_file(filepath + '/webpage' + str(step_now) + '.png') # 获取整个网页截图

            if step_now == 0:
                num = 1
                arr1[1].env=env
            else:
                num = 1

            step_IP = -float('inf')
            step_E = -float('inf')
            for i in range(1, num+1):
                # print("arr1[" + str(i) + "] = \n")
                # print(arr1[i])
                for j in range(1, N+1):
                    print("\nnode %d sub-node %d" % (i, j))
                    env_copy = copy.copy(arr1[i].env)
                    # env_copy = copy.deepcopy(arr1[i].env.copy())
                    # url_copy = env_copy.get_url()
                    # print("url_copy = ", url_copy)
                    # print("goto")
                    print("url = ", url)
                    env_copy.goto(url)
                    env_copy.steps = step_now
                    observation_copy = env_copy.observation()
                    # print("observation_copy = \n", observation_copy)
                    # 从当前步骤出发执行可能的下一步
                    action, reason = self.predict_action(
                        objective=objective, observation=observation_copy, url=url
                    )
                    print("action_content = ", action_trans(action, observation_copy))
                    screenfile = filepath + '/webpage' + str(step_now + 1) + '_' + str(i) + '_' + str(j) + '.png'
                    print("screenfile = ", screenfile)
                    status = env_copy.step2(action, screenfile) # 有可能第一步就完成了任务
                    # status = env.step(action)
                    # print("000subnode step_now = ", env_copy.steps)

                    arr2[(i - 1) * N + j].env = env_copy
                    arr2[(i - 1) * N + j].action = action
                    arr2[(i - 1) * N + j].reason = reason
                    arr2[(i - 1) * N + j].status = status
                    arr2[(i - 1) * N + j].AC = arr1[i].AC
                    arr2[(i - 1) * N + j].subnode = j

                    # 从下一步开始模拟
                    if not env_copy.done():
                        print("\nstart rollout")
                        while not env_copy.done():
                            print("subnode%d step_now = %d" % (j, env_copy.steps + 1))
                            # time.sleep(1)
                            observation_copy = env_copy.observation()
                            # print("\nobservation_copy = \n", observation_copy)
                            url_copy = env_copy.get_url()
                            print("url_copy = ", url_copy)
                            # 获取预测的行动和推理原因
                            action, reason = self.predict_action(
                                objective=objective, observation=observation_copy, url=url_copy
                            )
                            print("action_content = ", action_trans(action, observation_copy))
                            status = env_copy.step(action)

                            if (env_copy.steps > all_length):
                                break

                    # if not ((i==num) and (j==N)):
                    env_copy.is_done = False

                    if (env_copy.steps > all_length):
                        print("node%d subnode%d step length too long!" % (i, j))
                        continue

                    arr2[(i - 1) * N + j].success = status['success']
                    arr2[(i - 1) * N + j].TC = (1 - arr2[(i - 1) * N + j].AC) / (all_length - step_now - 1 + 1) * (1 - 2 * (1 - arr2[(i - 1) * N + j].success))
                    arr2[(i - 1) * N + j].AC = max(arr2[(i - 1) * N + j].AC, arr2[(i - 1) * N + j].AC + arr2[(i - 1) * N + j].TC)
                    print("arr2[%d].TC = %f" % ((i - 1) * N + j, arr2[(i - 1) * N + j].TC))
                    print("arr2[%d].AC = %f" % ((i - 1) * N + j, arr2[(i - 1) * N + j].AC))

                    if(status['success']):
                        print("node%d subnode%d success" % (i, j))
                        arr2[(i - 1) * N + j].length = status['num_actions']
                        # print("length = ", arr2[(i - 1) * N + j].length)
                        num_success += 1

                        # if (all_length == 0):
                        #     print("all_length = ", all_length)
                        #     all_length = arr2[(i - 1) * N + j].length

                        sum_length = sum_length + (arr2[(i - 1) * N + j].length - step_now - 1)
                        print("sum_length = ", sum_length)
                        if arr2[(i - 1) * N + j].length < min_length:
                            min_length = arr2[(i - 1) * N + j].length

                        if arr2[(i - 1) * N + j].length > max_length:
                            max_length = arr2[(i - 1) * N + j].length
                    print("\n")

                if(num_success):
                    print("num_success = ", num_success)
                    arr1[i].IP = num_success / N
                    # arr1[i].E = min_length/max_length
                    # if(step_now == 0):
                        # print("sum_length000 = ", sum_length)
                        # avg_length = sum_length / num_success
                        # all_length = avg_length
                        # print("avg_length000 = ", avg_length)
                        # print("all_length = ", all_length)
                        # arr1[i].E = 1 / all_length
                    # else:
                    print("sum_length = ", sum_length)
                    avg_length = sum_length / num_success
                    print("avg_length = ", avg_length)
                    print("last_length = ", last_length)
                    arr1[i].E = (last_length - avg_length) / all_length

                print("\narr1[%d].IP = %f" %(i, arr1[i].IP))
                print("arr1[%d].E = %f\n" % (i, arr1[i].E))
                step_IP = max(step_IP, arr1[i].IP)
                step_E = max(step_E, arr1[i].E)

            sorted(arr2)

            for i in range(1, N+1):
                arr1[i] = arr2[i]

            step_TC = arr1[1].TC

            last_length = avg_length

            env = arr1[1].env
            env.steps = step_now
            observation_copy = env.observation()
            # url = env.get_url()
            # print("final url = ", url)
            # screenfile = filepath + '/webpage' + str(step_now + 1) + '.png'
            # print("screenfile = ", screenfile)
            # env.goto2(url, screenfile)
            action = arr1[1].action
            reason = arr1[1].reason
            status = arr1[1].status
            print("actual step action")
            print("actual action = ", action)
            action_content = action_trans(action, observation_copy)
            print("actual action_content = ", action_content)

            env.step(action) # 选择了最佳的下一步后，执行这一步
            print("actual steps = ", env.steps)
            url = env.get_url()
            print("url2222 = %s\n" % url)

            # 删除没有选择的节点的图片
            choose_subnode = arr1[1].subnode
            print("choose_subnode = ", choose_subnode)
            print("filepath = ", filepath)
            for file in os.listdir(filepath):
                print("file1 = ", file)
                if (file.find('webpage') >= 0):
                    # print(file.find('webpage'))
                    # print(file[file.find('webpage') + len('webpage') : file.find('_')])
                    if (int(file[file.find('webpage') + len('webpage') : file.find('_')]) == env.steps):
                        print("file2 = ", file)
                        if (int(file[file.find('.png') - 1]) != choose_subnode):
                            os.remove(filepath + '/' + file)

            TR_C = self.get_TR_C(objective, step_now + 1, observation_copy, last_action, action_content)
            pos1 = TR_C.find("**Task Relevance**: ") + len("**Task Relevance**: ")
            pos2 = TR_C.find("**Coherence**: ") + len("**Coherence**: ")
            # print(TR_C[pos1 : pos1 + 1])
            step_TR = int(TR_C[pos1 : pos1 + 1])
            step_C = int(TR_C[pos2 : pos2 + 1])

            last_action = action_content

            print("\nstep_now : ", step_now + 1)
            print("action = ", action)
            print("step_IP = %f" % step_IP)
            print("step_E = %f" % step_E)
            print("step_TC = %f" % step_TC)
            print("step_TR = %f" % step_TR)
            print("step_C = %f\n" % step_C)

            now = {}
            temp_dict = {}
            temp_dict['step_idx'] = step_now + 1
            temp_dict['action'] = action_content # 转换成内容
            temp_dict['reason'] = reason
            temp_dict['status'] = status
            temp_dict['IP'] = step_IP
            temp_dict['E'] = step_E
            temp_dict['TC'] = step_TC
            temp_dict['TR'] = step_TR
            temp_dict['C'] = step_C

            now[step_now + 1] = temp_dict

            filename = filepath + '/evaluation_score.json'
            # print("filename = ", filename)

            if(os.path.exists(filename)):
                with open(filename, 'r') as f:
                    content = json.load(f)

                content.update(now)

                with open(filename, 'w') as f_new:
                    json.dump(content, f_new)
            else:
                with open(filename, 'w') as f:
                    json.dump(now, f)

            # if (num_success <= 0) and (step_now <= 1): # 剪枝
            #     break

            if "stop" in action:
                env.is_done = True

            if step_now >= self.max_actions:
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
