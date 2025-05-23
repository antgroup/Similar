import os
import time

os.environ[
    "SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "https://webarena-env-reddit.awsdev.asapp.com"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023/"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site


from webagents_step.environment.env import WebEnvironment
import json
import re
# Init an environment
from browser_env import (
    create_id_based_action,
    StateInfo,
    Trajectory,
    ActionTypes,
    ScriptBrowserEnv
)
from evaluation_harness.evaluators import evaluator_router


class WebArenaEnvironmentWrapper(WebEnvironment):
    def __init__(self, config_file, max_browser_rows=300, max_steps=50, slow_mo=1, observation_type="accessibility_tree", current_viewport_only=False, viewport_size={"width": 1280, "height": 720}, headless=False):
        self.webarena_env = ScriptBrowserEnv(
                    headless=headless,
                    slow_mo=slow_mo,
                    observation_type=observation_type,
                    current_viewport_only=current_viewport_only,
                    viewport_size=viewport_size
                )
        # print("555555555")
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        # filepath = "generate_data/" + self.config_file[:-5] + '/webpage0.png'
        # print("filepath = ", filepath)
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file}) # 打开网页
        self.terminated = False
        self.objective = self.config["intent"]
        self.url = self.config["start_url"]
        self.max_browser_rows = max_browser_rows
        self.max_steps = max_steps
        self.steps = 0
        self.is_done = False
        self.reward = 0.0
        self.action_limit_exceeded = False
        
        self.trajectory: Trajectory = []
        self.update_webarena_metrics()
        
    def reset(self):
        print("\nreset!!\n")
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})

    def close(self):
        self.webarena_env.close()

    def goto_ano(self, url):
        self.webarena_env.goto_ano(url)

    def goto(self, url):
        self.webarena_env.goto(url)

    def goto2(self, url, screenfile):
        self.webarena_env.goto2(url, screenfile)
        
    def get_url(self):
        return self.url
    
    def get_objective(self):
        return self.objective 
        
    def observation(self): 
        self.obs = self.webarena_env._get_obs()
        self.url = self.webarena_env.page.url
        browser_content = self.obs["text"]
        browser_content = browser_content.split("\n")[:self.max_browser_rows] 
        browser_content = "\n".join(browser_content)
        return browser_content
    
    def done(self):
        if self.is_done:
            return True
        return False
    
    def status(self):
        return {'done': self.is_done, 'reward': self.reward, 'success': float(self.reward > 0), 'num_actions': self.steps, 'action_limit_exceeded': self.action_limit_exceeded}

    def step(self, action):
        print("webarena step")
        self.steps = self.steps + 1
        print("\n")
        print(f"[Step {self.steps}] {action}")
        print("\n")

        if self.steps > self.max_steps:
            print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
            self.action_limit_exceeded = True
            self.is_done = True
            # action_cmd = create_id_based_action("stop [N/A]")
            action_cmd = create_id_based_action("stop maximum")
            self.update_webarena_metrics(action_cmd)
            return self.status()
        
        if "stop [" in action:
            # print("stop [")
            self.is_done = True # 1111
            action = action.replace('\\', '') 

        if action is None or action is "" or ("note [" in action):
            action_cmd = None
        else:
            try:
                action_cmd = create_id_based_action(action)
            except Exception as e:
                action_cmd = None

        if action_cmd:
            # print("action_cmd = ", action_cmd)
            # try:
                # time.sleep(2)
            self.obs, _, self.terminated, _, self.info = self.webarena_env.step(action_cmd)
            self.update_webarena_metrics(action_cmd)
            # except Exception as e:
            #     print(f"Error occurred while taking step: {e}")
            
        return self.status()

    def screenshot(self, screenfile):
        self.webarena_env.screenshot(screenfile)

    def step2(self, action, screenfile):
        # print("webarena step2")
        self.steps = self.steps + 1
        print("\n")
        print(f"[Step {self.steps}] {action}")
        print("\n")

        if self.steps > self.max_steps:
            print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
            self.action_limit_exceeded = True
            self.is_done = True
            # action_cmd = create_id_based_action("stop [N/A]")
            action_cmd = create_id_based_action("stop maximum")
            self.update_webarena_metrics(action_cmd)
            return self.status()

        if "stop [" in action:
            # print("stop [")
            self.is_done = True
            action = action.replace('\\', '')

        if action is None or action is "" or ("note [" in action):
            action_cmd = None
        else:
            action_cmd = create_id_based_action(action)

        if action_cmd:
            print("action_cmd = ", action_cmd)
            try:
                self.obs, _, self.terminated, _, self.info = self.webarena_env.step2(action_cmd, screenfile)
                self.update_webarena_metrics(action_cmd)
            except Exception as e:
                print(f"Error occurred while taking step: {e}")

        return self.status()

    
    def update_webarena_metrics(self, action_cmd=None):
        # Append action (if any) and resulting sate
        if action_cmd:
            # print("update_webarena_metrics action_cmd")
            self.trajectory.append(action_cmd)
            if action_cmd["action_type"]== ActionTypes.STOP: # 已经结束
                self.is_done = True

        if not self.is_done: # If we are done, no need to append state
            # print("update_webarena_metrics not done")
            state_info: StateInfo = {"observation": self.obs, "info": self.info}
            self.trajectory.append(state_info)
            
        if self.is_done:    
            try:
                evaluator = evaluator_router(self.config_file)
                self.reward = evaluator(trajectory=self.trajectory, config_file=self.config_file,
                                        page=self.webarena_env.page)
                # self.reward = evaluator(trajectory=self.trajectory, config_file=self.config_file, page=self.webarena_env.page, client=self.webarena_env.get_page_client(self.webarena_env.page))
            except Exception as e:
                print(f"Got excepetion: {e}")
                self.reward = 0