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

    while 'resultMap' not in response.text:
        response = requests.post(url, headers=headers, data=json.dumps(data))

    data = json.loads(response.text)
    score_map = data['resultMap']['objectAttributes']
    return score_map


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
        print("send error")
        print(e)
        return False

    try:
        return ask_chatgpt_async_fetch(param)
    except Exception as e:
        print("fetch error")
        print(e)
        return None


def get_TR_C(objective, steps, action_last, action):
    logger.info("get_TR_C")
    now_prompt = get_TR_C_prompt.replace("{Instruction}", objective)
    now_prompt = now_prompt.replace("{Steps}", str(steps))
    # now_prompt = now_prompt.replace("{Observation}", observation)
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
        return self.IP + self.E + self.TC + self.TR + self.C > other.IP + other.E + other.TC + other.TR + other.C


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


        for j in range(1, N + 1):
            logger.info("\nstep %d sub-node %d --------------------------------------------"% (step_now + 1, j))
            # 重置agent
            agent.reset()
            obs = env.reset(task_config=example)

            filepath2 = filepath + '-' + str(step_now + 1) + '-' + str(j)
            print("filepath2 = ", filepath2)
            if not os.path.isdir(filepath2):  # 如果没有该文件则新建一个
                os.makedirs(filepath2)

            instruction = goal
            trajectory = "Let\'s complete the task step by step. \n\n"

            action_id = 0

            done = False
            step_idx = 0

            # 执行到第n步
            logger.info("\nrecall action_list")
            for actions in action_list:
                action_id += 1
                print("\nactions = \n", actions)
                for action in actions:
                    obs, reward, done, info = env.step(action, args.sleep_after_execution)
                step_idx += 1
                trajectory += "Step " + str(action_id) + " : " + actions + "\n\n"
            step_idx = step_now

            screenfile = filepath + '/webpage' + str(step_now + 1) + '.png' # 主轨迹的页面截图
            print("screenfile = ", screenfile)
            if not os.path.exists(screenfile):
                with open(screenfile, "wb") as _f:
                    _f.write(obs['screenshot'])

            screenfile_sub = filepath2 + '/webpage' + str(step_now + 1) + '.png'  # 主轨迹的页面截图
            with open(screenfile, "wb") as _f:
                _f.write(obs['screenshot'])

            # screen_url_file = filepath + '/webpage' + str(step_now + 1) + 'url.txt'
            # observation_url = ''
            # if not os.path.exists(screen_url_file):
            #     resp = requests.post(url, data={"token": "5c7a6de660d7427ea66cc927a6b8a057"},
            #                          files={"file": open(screenfile, "rb")})
            #     observation_url = resp.json()['url']
            # else:
            #     with open(screen_url_file, "r", encoding='utf-8') as f:
            #         observation_url = f.readline()
            # print("observation_url = ", observation_url)
            observation_url = ''

            logger.info("\nDo right now step!")
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

            action_now = arr2[j].action
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

            done = False


        # 对arr2进行排序
        sorted(arr2)

        # 确定step_TC的值
        step_IP = arr2[1].IP
        step_E = arr2[1].E
        step_TC = arr2[1].TC
        step_TR = arr2[1].TR
        step_C = arr2[1].C

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

    # 保存每个任务的最终结果
    result_file = filepath + '/result.json'
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            json.dump(results, f)

    # 结束保存视频
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
