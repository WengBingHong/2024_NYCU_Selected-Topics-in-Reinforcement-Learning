from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
from collections import deque

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
        N_frame=8,
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
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=float32
        )
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert (
            self.camera_name in observation_spaces
        ), f"One of the sensors must be {self.camera_name}. Check the scenario file."
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 128, 128), dtype=np.uint8
        )
        #
        self.cur_step = 0
        self.frames = deque(maxlen=N_frame)

    def observation_postprocess(self, obs):

        obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)
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
        # frame stacking
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
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

        # obs, rewards, terminated, truncated, states 可能有這些東西
        obs, *others = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)

        # frame stacking
        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

        return obs, *others

    def render(self):
        return self.env.render()
