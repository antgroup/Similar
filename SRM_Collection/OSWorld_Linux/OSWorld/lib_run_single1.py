import datetime
import json
import logging
import os

from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")

# 运行单个样本
def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    # 重置agent和env
    agent.reset()
    obs = env.reset(task_config=example)

    done = False
    step_idx = 0
    # env.controller.start_recording()


    while not done and step_idx < max_steps:

        response, actions = agent.predict(
            instruction,
            obs
        )

        for action in actions:
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)

            # 保存屏幕截图以及轨迹信息
            with open(os.path.join(example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"),
                      "wb") as _f:
                _f.write(obs['screenshot'])

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png"
                }))
                f.write("\n")

            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    # 记录结果
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    # 结束保存视频
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))
