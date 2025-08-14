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
        if self.success >= other.success:
            if (self.success == other.success):
                return self.TC > other.TC
        else:
            return False



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

    task.tear_down(agent.env)

    N = 8
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

    all_length_filename = filepath + '/all_length.txt'

    if not os.path.exists(all_length_filename):
        sum_length0 = 0
        num_success0 = 0

        output = []
        # 从初识节点先模拟几个点，得到all_length
        for j in range(1, N + 1):
            print("\nnode 0 sub-node %d\n" % j)

            # 重置agent
            agent.reset(start_on_home_screen)
            task.initialize_task(agent.env)

            filepath2 = filepath + '-' + str(0) + '-' + str(j)
            print("\nfilepath2 = %s\n" % filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            step_idx = 0

            for step_idx in range(max_n_steps):
                screenfile = filepath2 + '/webpage' + str(step_idx + 1) + '.png'
                # print("\nscreenfile = %s\n" % screenfile)
                agent.screenshot(screenfile)

                result = agent.step(goal)
                action_content = agent.trans_action(result.data['action'], result.data['summary_origin'])
                reason = result.data['action_reason']

                print("\naction_now = ", action_content)
                print("\n")

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = step_idx + 1
                temp_dict['action'] = action_content
                temp_dict['reason'] = reason
                # 简化指标计算
                temp_dict['IP'] = 0
                temp_dict['E'] = 0
                temp_dict['TC'] = 0
                temp_dict['TR'] = 0
                temp_dict['C'] = 0

                now[str(step_idx + 1)] = temp_dict

                filename_now = filepath2 + '/evaluation_score.json'

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

                print('Completed step {:d}.'.format(step_idx + 1))
                print("\n")
                assert constants.STEP_NUMBER not in result.data

                if termination_fn(agent.env.controller):
                    print('Environment ends episode.\n')
                    break
                elif result.done:
                    print('Agent indicates task is done.\n')
                    break

            task_successful = task.is_successful(agent.env)  # 任务是否成功
            agent_successful = task_successful if result.done else 0.0  # 计算任务是否成功的指标
            success = 1 if agent_successful > 0.5 else 0

            task.tear_down(agent.env)

            print("task_successful = ", task_successful)
            print("agent_successful = ", agent_successful)
            print("success = ", success)

            # 如果当前模拟的节点能够完成任务
            if (success):
                print("\nnode 0 subnode%d success\n" % j)
                num_success0 += 1
                sum_length0 = sum_length0 + step_idx + 1
                print("sum_length0 = ", sum_length0)
                if os.path.exists(filepath2):
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)
            else:
                print("\nnode 0 subnode%d false\n" % j)

        # 如果模拟的节点都不能完成任务，则跳过这个任务
        if (num_success0 == 0):
            print("pass!!!")
            all_length = 0
            with open(all_length_filename, 'w') as f:
                f.write(str(all_length))

            output.append({constants.STEP_NUMBER: max_n_steps})

            task.initialize_task(agent.env)
            return EpisodeResult(done=False, step_data=_transpose_lod_to_dol(output))

        # 计算all_length
        print("sum_length0 = ", sum_length0)
        print("num_success0 = ", num_success0)
        all_length = sum_length0 / num_success0
        # all_length = 10
        print("all_length = %f\n" % all_length)

        with open(all_length_filename, 'w') as f:
            f.write(str(all_length))

        last_length = all_length
    else:
        print("all_length exists!")
        with open(all_length_filename, "r", encoding='utf-8') as f:
            all_length = float(f.readline())
        print("all_length = %f\n" % all_length)
        if all_length == 0:
            print("all_length == 0 pass!!!")

            output = []
            output.append({constants.STEP_NUMBER: max_n_steps})

            task.initialize_task(agent.env)
            return EpisodeResult(done=False, step_data=_transpose_lod_to_dol(output))

        last_length = all_length

    print("pre-action is over------------------------------------\n")


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
        if (step_now > all_length + 1):
            print("node step length too long!")
            break

        suc = [0 for _ in range(N + 1)]

        # 初始化当前步骤计算需要的值
        num_success = 0
        sum_length = 0
        step_IP = 0
        step_E = 0

        for j in range(1, N + 1):
            print("\nstep %d sub-node %d --------------------------------------------\n" % (step_now + 1, j))
            # 重置agent
            agent.reset(start_on_home_screen)
            task.initialize_task(agent.env)
            # task.initialize_task(agent.env)

            # 执行到第n步
            print("\nrecall action_list")
            for action_dict in action_list:
                action = action_dict['action']
                reason = action_dict['reason']
                print("\naction = ", action)
                print("reason = \n", reason)
                agent.execute(goal, action, reason)

            screenfile = filepath + '/webpage' + str(step_now + 1) + '.png'
            print("\nscreenfile = %s\n" % screenfile)
            agent.screenshot(screenfile)

            print("\nDo right now step!")
            # 执行当前步骤的操作
            result = agent.step(goal)

            # 存储的是执行动作后的状态
            arr2[j].result = result
            arr2[j].action = agent.trans_action(result.data['action'], result.data['summary_origin'])
            arr2[j].reason = result.data['action_reason']
            arr2[j].subnode = j

            print("action_now = ", arr2[j].action)

            # 可能这一步会完成任务
            print('Completed step {:d}.'.format(step_now + 1))
            print("\n")
            assert constants.STEP_NUMBER not in result.data

            if termination_fn(agent.env.controller):
                task.tear_down(agent.env)
                print('Environment ends episode.\n')
                break
            elif result.done:
                task.tear_down(agent.env)
                print('Agent indicates task is done.\n')
                break

            # 从下一步开始模拟
            print("\nstart rollouting-------------------------------------------")
            filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
            print("filepath2 = ", filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            filename_now = filepath2 + '/evaluation_score.json'
            filename = filepath + '/evaluation_score.json'
            if not os.path.exists(filename_now):
                now = {}
                with open(filename_now, 'w') as f:
                    json.dump(now, f)
                if os.path.exists(filename):
                    shutil.copyfile(filename, filename_now)

            step_idx = step_now + 1

            for step_idx in range(step_now + 1, max_n_steps):
                screenfile = filepath2 + '/webpage' + str(step_idx + 1) + '.png'
                print("screenfile = ", screenfile)
                agent.screenshot(screenfile)

                result = agent.step(goal)
                action_content = agent.trans_action(result.data['action'], result.data['summary_origin'])
                reason = result.data['action_reason']

                print("\naction_now = ", action_content)
                print("\n")

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = step_idx + 1
                temp_dict['action'] = action_content
                temp_dict['reason'] = reason
                # 简化指标计算
                temp_dict['IP'] = 1
                temp_dict['E'] = 1 / all_length
                temp_dict['TC'] = 1 / all_length
                temp_dict['TR'] = 1
                temp_dict['C'] = 1

                now[str(step_idx + 1)] = temp_dict

                filename_now = filepath2 + '/evaluation_score.json'

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

                print('Completed step {:d}.'.format(step_idx + 1))
                print("\n")
                assert constants.STEP_NUMBER not in result.data

                if termination_fn(agent.env.controller):
                    print('Environment ends episode.\n')
                    break
                elif result.done:
                    print('Agent indicates task is done.\n')
                    break

            task_successful = task.is_successful(agent.env)  # 任务是否成功
            agent_successful = task_successful if result.done else 0.0  # 计算任务是否成功的指标
            success = 1 if agent_successful > 0.5 else 0

            task.tear_down(agent.env)

            print("task_successful = ", task_successful)
            print("agent_successful = ", agent_successful)
            print("success = ", success)

            arr2[j].success = success
            # 计算节点的TC和AC值
            arr2[j].TC = (1 - arr2[j].AC) / (all_length - step_now - 1 + 1) * (1 - 2 * (1 - arr2[j].success))
            arr2[j].AC = max(arr2[j].AC, arr2[j].AC + arr2[j].TC)
            print("arr2[%d].TC = %f" % (j, arr2[j].TC))
            print("arr2[%d].AC = %f" % (j, arr2[j].AC))

            # 如果当前节点是成功的
            if (success):
                print("\nsubnode%d success\n" % (j))
                num_success += 1

                suc[j] = step_idx
                # print("suc[%d] = %d" % (j, suc[j]))

                arr2[j].length = step_idx
                # print("length = ", arr2[j].length)

                sum_length = sum_length + (arr2[j].length - step_now - 1)
                # print("sum_length = ", sum_length)
            else:
                suc[j] = max_n_steps
                print("\nsubnode%d false\n" % (j))
                # 没有成功的节点创建的文件夹直接删除
                if os.path.exists(filepath2):
                    print("subnode%d fail" % (j))
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)

            print("\n")

        if (num_success):
            print("\nstep_now %d success!!\n" % (step_now + 1))
            print("num_success = ", num_success)

            print("sum_length = ", sum_length)
            avg_length = sum_length / num_success
            print("avg_length = ", avg_length)
            print("last_length = ", last_length)

            step_IP = num_success / N
            step_E = (last_length - avg_length) / all_length
        else:
            avg_length = all_length

        last_length = avg_length  # 更新last_length的值

        # 对arr2进行排序
        sorted(arr2)

        # 确定step_TC的值
        step_TC = arr2[1].TC

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

        # 获得step_TR和step_C的值
        TR_C = agent.get_TR_C(goal, step_now + 1, last_action, action)
        pos1 = TR_C.find("**Task Relevance**: ") + len("**Task Relevance**: ")
        pos2 = TR_C.find("**Coherence**: ") + len("**Coherence**: ")
        step_TR = int(TR_C[pos1: pos1 + 1])
        step_C = int(TR_C[pos2: pos2 + 1])

        last_action = action

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

        # 完善衍生数据
        print("complete extra datas")
        for j in range(1, N + 1):
            filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
            if not os.path.exists(filepath2):
                continue

            dis = suc[j] - (step_now + 1) + 1
            # print("dis = ", dis)

            for k in range(1, step_now + 1 + 1):
                screenfile = filepath + '/webpage' + str(k) + '.png'
                screenfile_now = filepath2 + '/webpage' + str(k) + '.png'
                shutil.copyfile(screenfile, screenfile_now)

            filename = filepath2 + '/evaluation_score.json'

            if (os.path.exists(filename)):  # 如果已经存在文件
                with open(filename, 'r') as f:
                    content = json.load(f)
                content.update(now)
                content_new = dict(sorted(content.items(), reverse=False))  # 重新排序
                # 更新数据
                with open(filename, 'w') as f_new:
                    json.dump(content_new, f_new)
            else:  # 如果没有文件
                with open(filename, 'w') as f:
                    json.dump(now, f)

            # 更新E、TC的值
            with open(filename, 'r', encoding='UTF-8') as f:
                dict_now = json.load(f)
                content_now = dict_now

            for (key, value) in dict_now.items():
                temp = {}
                value_now = value
                value_now['E'] = 1 / dis
                value_now['TC'] = 1 / dis
                temp[key] = value_now
                content_now.update(temp)
            content_now_new = dict(sorted(content_now.items(), reverse=False))  # 重新排序
            with open(filename, 'w') as f_new:
                json.dump(content_now_new, f_new)

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
