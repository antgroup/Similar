# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs an agent on the environment."""

import dataclasses
import os
import json
import shutil
from typing import Any, Callable, Optional

from android_env import env_interface
from android_world import constants
from android_world.agents import base_agent
import termcolor

from android_world.task_evals import task_eval

import requests


url = "https://api.superbed.cn/upload"

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
            "_TIMEOUT_": "30000",
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
    data = json.loads(response.text)
    score_map = data['resultMap']['objectAttributes']
    return score_map



class Node:
    def __init__(self):
        self.env = None
        self.observation = None
        self.url = None
        self.step_now = 0
        self.subnode = 0

        self.result = None
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
        return self.IP + self.E + self.TC+ self.TR+ self.C> other.IP + other.E + other.TC+ other.TR+ other.C



@dataclasses.dataclass()
class EpisodeResult:
    """Represents an episode of an agent interacting with the environment.

    Attributes:
      done: Whether the agent indicated the task is complete.
      step_data: Environment and agent data for each step.
      env_reward: Reward returned by environment, if applicable.
    """

    done: bool
    step_data: dict[str, Any]
    env_reward: Optional[float] = None


def run_episode(
        task: task_eval.TaskEval,
        name: str,
        goal: str,
        agent: base_agent.EnvironmentInteractingAgent,
        max_n_steps: int = 10,
        start_on_home_screen: bool = False,
        termination_fn: (
                Callable[[env_interface.AndroidEnvInterface], float] | None
        ) = None,
) -> EpisodeResult:
    """Runs an agent on goal, e.g., "turn off wifi".

    An agent will start from whatever state the provided environment is in and
    run until it determines a task is complete, if the max number of
    steps is reached, of if the termination_fn is True.

    Args:
      task: The task.
      name: The name of the task.
      goal: The goal instruction for the agent.
      agent: The agent to run on the environment.
      max_n_steps: The max number of steps to allow an agent to run before ending
        an episode.
      start_on_home_screen: Whether to start episode from the home screen or just
        the current screen.
      termination_fn: If provided, a determines whether to terminate an episode.
        For example, for MiniWoB++ tasks, the episode should terminate if there is
        a nonzero reward.

    Returns:
      Data collected during running agent on goal.
    """
    if max_n_steps == 0:
        return EpisodeResult(done=False, step_data={})
    if termination_fn is None:
        termination_fn = lambda env: False

    print("\ngoal = %s\n" % goal)

    # task.tear_down(agent.env)

    N = 3
    sum = N * (N + 1)

    arr2 = [Node() for _ in range(sum + 1)]
    action_list = []

    # 生成的数据存储的位置
    filepath = "generate_data/" + name
    print("filepath : %s", filepath)
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
            temp_dict = {}
            temp_dict['action'] = value['action']
            temp_dict['reason'] = value['reason']
            action_list.append(temp_dict)
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


    avg_length = 0
    last_action = ''

    agent.reset(start_on_home_screen)
    task.tear_down(agent.env)

    output = []

    for step_now in range(step_exist, max_n_steps):
        print("\nstep_now : %d ----------------------------------------------------------------------" % (step_now + 1))


        for j in range(1, N + 1):
            print("\nstep %d sub-node %d --------------------------------------------\n" % (step_now + 1, j))
            # 重置agent
            agent.reset(start_on_home_screen)
            task.initialize_task(agent.env)

            filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
            print("filepath2 = ", filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            instruction = goal
            trajectory = "Let\'s complete the task step by step. \n\n"

            action_id = 0

            # 执行到第n步
            print("\nrecall action_list")
            for action_dict in action_list:
                action_id += 1
                action = action_dict['action']
                reason = action_dict['reason']
                print("\naction = ", action)
                print("reason = \n", reason)
                agent.execute(goal, action, reason)
                trajectory += "Step " + str(action_id) + " : " + action + "\n\n"

            screenfile = filepath + '/webpage' + str(step_now + 1) + '.png' # 主轨迹的页面截图
            print("\nscreenfile = %s\n" % screenfile)
            if not os.path.exists(screenfile):
                agent.screenshot(screenfile)

            screenfile_sub = filepath2 + '/webpage' + str(step_now + 1) + '.png'  # 主轨迹的页面截图
            agent.screenshot(screenfile_sub)

            screen_url_file = filepath + '/webpage' + str(step_now + 1) + 'url.txt'  # 主轨迹的页面截图
            observation_url = ''
            if not os.path.exists(screen_url_file):
                resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"},
                                     files={"file": open(screenfile, "rb")})
                observation_url = resp.json()['url']
            else:
                with open(screen_url_file, "r", encoding='utf-8') as f:
                    observation_url = f.readline()
            print("observation_url = ", observation_url)


            print("\nDo right now step!")
            # 执行当前步骤的操作
            result = agent.step(goal)

            # 存储的是执行动作后的状态
            arr2[j].result = result
            arr2[j].action = agent.trans_action(result.data['action'], result.data['summary_origin'])
            arr2[j].reason = result.data['action_reason']
            arr2[j].subnode = j

            action_now = arr2[j].action
            reason_now = arr2[j].reason
            print("action_now = ", arr2[j].action)

            score_map = get_similar_score(instruction, trajectory, observation_url, action_now)
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
            temp_dict['action'] = action_now
            temp_dict['reason'] = reason_now

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

            # 可能这一步会完成任务
            print('Completed step {:d}.'.format(step_now + 1))
            print("\n")

            task.tear_down(agent.env)


        # 对arr2进行排序
        sorted(arr2)

        # 确定step_TC的值
        step_IP = arr2[1].IP
        step_E = arr2[1].E
        step_TC = arr2[1].TC
        step_TR = arr2[1].TR
        step_C = arr2[1].C

        # 确定当前步骤的action
        action = arr2[1].action # 转换之后的action
        reason = arr2[1].reason
        result = arr2[1].result
        print("\nactual action = ", action)
        print("actual reason = \n", reason)

        # 更新action_list
        action_now = {}
        temp_dict = {}
        temp_dict['action'] = result.data['action'] # 原本的action
        temp_dict['reason'] = reason
        action_list.append(temp_dict)  # 当前程序要跑的action_list
        action_now[step_now + 1] = temp_dict
        if (os.path.exists(action_list_file)):  # 如果已经存在文件
            with open(action_list_file, 'r') as f:
                content = json.load(f)
            content.update(action_now)
            # 更新数据
            with open(action_list_file, 'w') as f_new:
                json.dump(content, f_new)


        # 总结并存储当前步骤的数据
        print("step_now : %d", step_now + 1)
        print("action = \n", action)
        print("step_IP = %f" % step_IP)
        print("step_E = %f" % step_E)
        print("step_TC = %f" % step_TC)
        print("step_TR = %f" % step_TR)
        print("step_C = %f\n" % step_C)

        now = {}
        temp_dict = {}
        temp_dict['step_idx'] = step_now + 1
        temp_dict['action'] = action
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



        print('Completed step {:d}.'.format(step_now + 1))
        print("\n")
        assert constants.STEP_NUMBER not in result.data
        output.append(result.data | {constants.STEP_NUMBER: step_now})

        if termination_fn(agent.env.controller):
            print('Environment ends episode.\n')
            task.initialize_task(agent.env)
            return EpisodeResult(
                done=True,
                step_data=_transpose_lod_to_dol(output),
            )
        elif result.done:
            print('Agent indicates task is done.\n')
            task.initialize_task(agent.env)
            return EpisodeResult(
                done=result.done,
                step_data=_transpose_lod_to_dol(output),
            )

    print(
        termcolor.colored(
            'Agent did not indicate task is done. Reached max number of steps.',
            'red',
        )
    )
    output = []
    output.append({constants.STEP_NUMBER: max_n_steps})

    task.initialize_task(agent.env)
    return EpisodeResult(done=False, step_data=_transpose_lod_to_dol(output))
    # task.initialize_task(agent.env)
    # return EpisodeResult(
    #     done=result.done, step_data=_transpose_lod_to_dol(output)  # pylint: disable=undefined-variable
    # )


def _transpose_lod_to_dol(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Transposes a list of dictionaries to a dictionary of lists.

    Args:
      data: A list of dictionaries.

    Returns:
      A dictionary where each key is from the input dictionaries and each value is
      a list of values for that key.
    """
    result = {}
    for d in data:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result


def transpose_dol_to_lod(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Converts a dictionary of lists to a list of dictionaries.

    Useful for post-processing of results; e.g., in colab.

    Args:
      data: A dictionary where each value is a list.

    Returns:
      A list of dictionaries.
    """
    return [dict(zip(data.keys(), values)) for values in zip(*data.values())]
