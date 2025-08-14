import datetime
import json
import logging
import os
import shutil

from wrapt_timeout_decorator import *
from utils.utils import (
    get_default_config,
    ask_chatgpt_async_send,
    ask_chatgpt_async_fetch
)

logger = logging.getLogger("desktopenv.experiment")

get_TR_C_prompt = '''
You are an expert in evaluating the performance of a Virtual Agent. The Virtual Agent is designed to help a human user complete specified tasks (such as web navigation, web content Q&A, etc.) on various platform applications (such as websites, mobile apps, etc.). Given the user’s goal, the current step number, the agent’s last action (if there exists) and the agent’s current action, your goal is to evaluate the quality of the current step's action of the agent. Please evaluate the quality of the current step's action of the agent based on the given dimensions.
<Goal>: {Instruction}
<Current Step Number>: {Steps}
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


def ask_chatgpt(model, msg, temp='0.2'):
    # print("model = ", model)
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


def get_TR_C(objective, steps, observation, action_last, action):
    logger.info("get_TR_C")
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
    # print("msg = ", msg)

    flag = 1

    while flag:
        # sleep(0.5)
        response = ask_chatgpt(model, msg, temperature_str)
        if response != None and response != False:
            flag = 0
            # print("yes111")
            print("\nresponse = \n", response)
        else:
            flag = 1

    return response


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


# 运行单个样本
def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores, example_id):
    N = 3
    sum = N * (N + 1)

    arr2 = [Node() for _ in range(sum + 1)]
    action_list = []

    # 生成的数据存储的位置
    filepath = "generate_data/" + example_id
    logger.info("filepath : %s", filepath)
    if not os.path.isdir(filepath):  # 如果没有该文件则新建一个
        os.makedirs(filepath)

    action_list_file = filepath + '/action_list.json'
    if not os.path.exists(action_list_file):
        now = {}
        with open(action_list_file, 'w') as f:
            json.dump(now, f)

    all_length_filename = filepath + '/all_length.text'

    if not os.path.exists(all_length_filename):
        sum_length0 = 0
        num_success0 = 0

        # 从初识节点先模拟几个点，得到all_length
        for j in range(1, N + 1):
            logger.info("\nnode 0 sub-node %d" % j)

            # 重置agent和env
            agent.reset()
            obs = env.reset(task_config=example)

            filepath2 = filepath + '-' + str(0) + '-' + str(j)
            print("filepath2 = ", filepath2)
            # print("filepath2 = ", filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            done = False
            step_idx = 0

            # rollout
            while not done:
                if (step_idx + 1 > max_steps):
                    print("\nnode 0 subnode%d steps exceeded!!\n", j)
                    break

                # print("subnode%d step_now = %d" % (j, step_idx + 1))

                screenfile = filepath2 + '/webpage' + str(step_idx + 1) + '.png'
                print("screenfile = ", screenfile)
                with open(screenfile, "wb") as _f:
                    _f.write(obs['screenshot'])

                response, actions = agent.predict(
                    instruction,
                    obs
                )

                for action in actions:
                    # print("-----------------------------------------------------------")
                    # print("action = \n", action)
                    # print("-----------------------------------------------------------")

                    # 记录当前操作
                    logger.info("node 0 subnode %d Step %d: \n%s", j, step_idx + 1, action)
                    # 执行当前操作，更新obs，获得reward、done、info
                    obs, reward, done, info = env.step(action, args.sleep_after_execution)

                    # 如果任务完成
                    if done:
                        break
                step_idx += 1

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = step_idx
                temp_dict['action'] = actions
                # 简化指标计算
                temp_dict['IP'] = 0
                temp_dict['E'] = 0
                temp_dict['TC'] = 0
                temp_dict['TR'] = 0
                temp_dict['C'] = 0

                now[str(step_idx)] = temp_dict

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

            result = env.evaluate()
            logger.info("Result: %.2f", result)

            # 如果当前模拟的节点能够完成任务
            if (result > 0):
                logger.info("node 0 subnode%d success\n" % j)
                num_success0 += 1
                sum_length0 = sum_length0 + step_idx
                if os.path.exists(filepath2):
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)
            else:
                logger.info("node 0 subnode%d false\n" % j)

        # 如果模拟的节点都不能完成任务，则跳过这个任务
        if (num_success0 == 0):
            print("pass!!!")
            all_length = 0
            with open(all_length_filename, 'w') as f:
                f.write(str(all_length))

            result = 0.0
            logger.info("Result: %.2f", result)
            scores.append(result)
            with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"{result}\n")
            return

        # 计算all_length
        print("sum_length0 = ", sum_length0)
        print("num_success0 = ", num_success0)
        all_length = sum_length0 / num_success0
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
        if all_length == 0:
            print("all_length == 0 pass!!!")

            result = 0.0
            logger.info("Result: %.2f", result)
            scores.append(result)
            with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
                f.write(f"{result}\n")
            return
        last_length = all_length

    logger.info("pre-action is over------------------------------------\n")

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
    last_action = ['']

    # 重置agent和env
    # agent.reset()
    # obs = env.reset(task_config=example)

    done = False
    # step_idx = 0
    step_idx = step_exist
    # env.controller.start_recording()

    while not done:
        if (step_idx + 1 > max_steps):
            print("\nsteps exceeded!!")
            print("task done\n")
            break

        step_now = step_idx
        logger.info("\nstep_now : %d ----------------------------------------------------------------------" % (step_now + 1))
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
            logger.info("\nstep %d sub-node %d --------------------------------------------"% (step_now + 1, j))

            agent.reset()
            obs = env.reset(task_config=example)

            done = False
            step_idx = 0

            # 执行到第n步
            logger.info("\nrecall action_list")
            for actions in action_list:
                print("\nactions = \n", actions)
                for action in actions:
                    obs, reward, done, info = env.step(action, args.sleep_after_execution)
                step_idx += 1
            step_idx = step_now

            screenfile = filepath + '/webpage' + str(step_now + 1) + '.png'
            print("screenfile = ", screenfile)
            if not os.path.exists(screenfile):
                with open(screenfile, "wb") as _f:
                    _f.write(obs['screenshot'])

            # 执行当前步骤的操作
            response, actions = agent.predict(
                instruction,
                obs
            )
            for action in actions:
                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                logger.info("step %d subnode %d: \n%s", step_now + 1, j, action)
            step_idx += 1

            # 存储的是执行动作后的状态
            arr2[j].env = env
            arr2[j].action = actions
            arr2[j].subnode = j

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

            while not done:
                if (step_idx + 1 > all_length):
                    break

                logger.info("step %d subnode %d step_idx = %d" % (step_now, j, step_idx + 1))

                screenfile = filepath2 + '/webpage' + str(step_idx + 1) + '.png'
                print("screenfile = ", screenfile)
                with open(screenfile, "wb") as _f:
                    _f.write(obs['screenshot'])

                response, actions = agent.predict(
                    instruction,
                    obs
                )
                for action in actions:
                    # 记录当前操作
                    logger.info("step %d subnode %d step %d: \n%s", step_now, j, step_idx + 1, action)
                    obs, reward, done, info = env.step(action, args.sleep_after_execution)

                    # 如果任务完成
                    if done:
                        break
                step_idx += 1

                # 存储步骤级数据
                now = {}
                temp_dict = {}
                temp_dict['step_idx'] = step_idx
                temp_dict['action'] = actions
                # 简化指标计算
                temp_dict['IP'] = 1
                temp_dict['E'] = 1 / all_length
                temp_dict['TC'] = 1 / all_length
                temp_dict['TR'] = 1
                temp_dict['C'] = 1

                now[str(step_idx)] = temp_dict

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

            result = env.evaluate()
            logger.info("Result: %.2f", result)

            if (step_idx + 1 > all_length):
                logger.info("subnode%d false" % (j))
                print("subnode%d step length too long!" % (j))
                if os.path.exists(filepath2):
                    print("remove filepath2 = ", filepath2)
                    shutil.rmtree(filepath2)
                continue

            arr2[j].success = result
            # 计算节点的TC和AC值
            arr2[j].TC = (1 - arr2[j].AC) / (all_length - step_now - 1 + 1) * (1 - 2 * (1 - arr2[j].success))
            arr2[j].AC = max(arr2[j].AC, arr2[j].AC + arr2[j].TC)
            print("arr2[%d].TC = %f" % (j, arr2[j].TC))
            print("arr2[%d].AC = %f" % (j, arr2[j].AC))

            # 如果当前节点是成功的
            if (result > 0):
                logger.info("subnode%d success" % (j))
                num_success += 1

                suc[j] = step_idx

                arr2[j].length = step_idx
                # print("length = ", arr2[j].length)

                sum_length = sum_length + (arr2[j].length - step_now - 1)
                # print("sum_length = ", sum_length)
            else:
                logger.info("subnode%d false" % (j))
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

        step_idx = step_now + 1

        # 确定当前步骤的action
        actions = arr2[1].action
        print("actual action = \n", actions)

        # 更新action_list
        action_list.append(actions)
        action_now = {}
        temp_dict = {}
        temp_dict['action'] = actions
        action_now[step_now + 1] = temp_dict
        if (os.path.exists(action_list_file)):  # 如果已经存在文件
            with open(action_list_file, 'r') as f:
                content = json.load(f)
            content.update(action_now)
            # 更新数据
            with open(action_list_file, 'w') as f_new:
                json.dump(content, f_new)

        # 获得step_TR和step_C的值
        TR_C = get_TR_C(instruction, step_now + 1, obs['accessibility_tree'], "'" + last_action[0] + "'", "'" + actions[0] + "'")
        pos1 = TR_C.find("**Task Relevance**: ") + len("**Task Relevance**: ")
        pos2 = TR_C.find("**Coherence**: ") + len("**Coherence**: ")
        step_TR = int(TR_C[pos1: pos1 + 1])
        step_C = int(TR_C[pos2: pos2 + 1])

        last_action = actions

        # 总结并存储当前步骤的数据
        logger.info("step_now : %d", step_now + 1)
        print("action = \n", actions)
        print("step_IP = %f" % step_IP)
        print("step_E = %f" % step_E)
        print("step_TC = %f" % step_TC)
        print("step_TR = %f" % step_TR)
        print("step_C = %f\n" % step_C)

        now = {}
        temp_dict = {}
        temp_dict['step_idx'] = step_now + 1
        temp_dict['action'] = actions
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
        logger.info("complete extra datas")
        for j in range(1, N + 1):
            filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
            if not os.path.exists(filepath2):
                continue

            dis = suc[j] - (step_now + 1) + 1
            print("dis = ", dis)

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


        if "DONE" in actions:
            done = True
            break

        if step_now >= max_steps:
            print(f"Agent exceeded max actions: {self.max_actions}")
            break


    # 记录结果
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    # 结束保存视频
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
