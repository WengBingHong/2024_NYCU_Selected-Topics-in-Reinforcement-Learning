from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
import cv2

# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api


class RaceEnv(gym.Env):
    camera_name = "camera_competition"
    motor_name = "motor_competition"
    steering_name = "steering_competition"
    """The environment wrapper for RaceCarGym.
    
    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'
    
    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """

    def __init__(
        self,
        scenario: str,
        render_mode: str = "rgb_array_birds_eye",
        reset_when_collision: bool = True,
        **kwargs,
    ):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f"SingleAgent{self.scenario}-v0"
        self.env = gym.make(
            id=self.env_id,
            render_mode=render_mode,
            reset_when_collision=reset_when_collision,
            **kwargs,
        )
        self.render_mode = render_mode
        # Assume actions only include: motor and steering # 改成只前進與右轉
        # self.action_space = gym.spaces.box.Box(
        #     low=0, high=1.0, shape=(2,), dtype=float32
        # )
        self.action_space = gym.spaces.Box(
            low=np.array(
                [1.0, 0.0], dtype=np.float32
            ),  # 第一个维度固定为 1.0，第二个维度最低为 0.0
            high=np.array(
                [1.0, 0.7], dtype=np.float32
            ),  # 第一个维度固定为 1.0，第二个维度最高为 0.7
            dtype=np.float32,
        )
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        # self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        self.cur_step = 0
        self.prev_info = dict()

    def observation_postprocess(self, obs):
        # obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1) # 從 128,128,3 轉換成 3,128,128

        obs = obs[self.camera_name].astype(np.uint8)  # 128,128,3
        # print("obs shape", obs.shape)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # 灰階化，形狀變為 (128, 128)
        # print("obs gray shape", obs.shape)
        obs = cv2.resize(
            obs, (84, 84), interpolation=cv2.INTER_AREA
        )  # 形狀變為 (84, 84)
        # print("obs resize shape", obs.shape)
        obs = np.expand_dims(obs, axis=0)  # 新形狀: (1, 84, 84)
        # print("obs expand shape", obs.shape)
        return obs

    def reset(self, *args, **kwargs: dict):

        # 設定random地圖出生點
        if kwargs.get("options"):
            kwargs["options"]["mode"] = "random"
        else:
            kwargs["options"] = {"mode": "random"}

        self.cur_step = 0
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.observation_postprocess(obs)
        self.prev_info["motor"] = 0
        self.prev_info["steering"] = 0
        self.prev_info["state"] = others[0].copy()
        return obs, *others

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # Add a small noise and clip the actions
        motor_scale = 0.001
        steering_scale = 0.01
        motor_action = np.clip(
            motor_action + np.random.normal(scale=motor_scale), -1.0, 1.0
        )
        steering_action = np.clip(
            steering_action + np.random.normal(scale=steering_scale), -1.0, 1.0
        )

        dict_actions = OrderedDict(
            [
                (self.motor_name, array(motor_action, dtype=float32)),
                (self.steering_name, array(steering_action, dtype=float32)),
            ]
        )

        obs, *others = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        reward, terminated, truncated, state = others

        # # 1st attempt:
        # reward = 0
        # reward += 1 * motor_action
        # reward -= 0.1 * (abs(motor_action - self.prev_info['motor']) + abs(steering_action - self.prev_info['steering']))
        # if state['progress'] > self.prev_info['state']['progress']: # move forward
        #     reward += 1000 * (state['progress'] - self.prev_info['state']['progress'])
        # elif state['progress'] == self.prev_info['state']['progress']:   # not moving
        #     reward -= 0.3
        # if state['wall_collision'] == True:
        #     reward = -100
        #     terminated = True
        # self.prev_info['motor'] = motor_action.copy()
        # self.prev_info['steering'] = steering_action.copy()
        # self.prev_info['state'] = state.copy()

        # # 2nd attempt:
        # reward = 0
        # reward += 1 * motor_action
        # reward -= 0.1 * (
        #     abs(motor_action - self.prev_info["motor"])
        #     + abs(steering_action - self.prev_info["steering"])
        # )
        # if state["progress"] > self.prev_info["state"]["progress"]:  # move forward
        #     reward += 1000 * (state["progress"] - self.prev_info["state"]["progress"])
        # elif state["progress"] == self.prev_info["state"]["progress"]:  # not moving
        #     reward -= 0.3
        # if state["wall_collision"] == True:
        #     reward = -300
        #     terminated = True
        # self.prev_info["motor"] = motor_action.copy()
        # self.prev_info["steering"] = steering_action.copy()
        # self.prev_info["state"] = state.copy()

        # 3rd attempt:
        # reward = 0
        # reward += 1 * motor_action
        # # reward -= 0.5 * (abs(steering_action - self.prev_info["steering"]))
        # if steering_action < 0:  # 往左扣分 鼓勵往右
        #     reward += 2 * steering_action
        # if state["progress"] > self.prev_info["state"]["progress"]:  # move forward
        #     reward += 1000 * (state["progress"] - self.prev_info["state"]["progress"])
        # elif state["progress"] == self.prev_info["state"]["progress"]:  # not moving
        #     reward -= 0.3
        # if state["wall_collision"] == True:
        #     reward = -500
        #     terminated = True
        # self.prev_info["motor"] = motor_action.copy()
        # self.prev_info["steering"] = steering_action.copy()
        # self.prev_info["state"] = state.copy()

        # 4th attempt:
        reward = 0
        reward += 1 * motor_action
        reward -= 0.5 * (abs(steering_action - self.prev_info["steering"]))
        # 往左扣分 鼓勵往右
        if steering_action < 0:
            reward += 3 * steering_action
        elif steering_action > 0.8:
            reward -= 3 * steering_action
        else:
            reward += steering_action

        if state["progress"] > self.prev_info["state"]["progress"]:  # move forward
            reward += 1000 * (state["progress"] - self.prev_info["state"]["progress"])
        elif state["progress"] == self.prev_info["state"]["progress"]:  # not moving
            reward -= 0.3
        if state["wall_collision"] == True:
            reward = -500
            terminated = True
        self.prev_info["motor"] = motor_action.copy()
        self.prev_info["steering"] = steering_action.copy()
        self.prev_info["state"] = state.copy()

        # obs, reward, terminated, truncated, state 可能有這些東西
        obs, *others = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        # return obs, *others
        return obs, reward, terminated, truncated, state

    def render(self):
        return self.env.render()
