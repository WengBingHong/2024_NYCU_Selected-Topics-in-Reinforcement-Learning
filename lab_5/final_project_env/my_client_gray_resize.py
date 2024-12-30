import argparse
import json
import numpy as np
import requests

import gymnasium as gym

from stable_baselines3 import DQN, PPO, SAC, TD3


def connect(agent, url: str = "http://localhost:5000"):
    while True:
        print("=========================Frame==========================")
        # Get the observation
        response = requests.get(f"{url}")
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break
        # Get observation from server
        obs = json.loads(response.text)["observation"]
        obs = np.array(obs).astype(np.uint8)

        # the shape is(3, 128, 128)
        print("np.array(obs).astype(np.uint8)")
        print(obs.shape)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action
        # action(motor, steering)
        print("action(motor, steering)")
        print(action_to_take)

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f"{url}", json={"action": action_to_take.tolist()})
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break

        print("response")
        print(response)

        result = json.loads(response.text)
        terminal = result["terminal"]

        print("the result")
        print(result)

        if terminal:
            print("Episode finished.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="The url of the server.",
    )
    args = parser.parse_args()

    # class RandomAgent:
    #     def __init__(self, action_space):
    #         self.action_space = action_space

    #     def act(self, observation):
    #         return self.action_space.sample()

    # # Initialize the RL Agent
    # import gymnasium as gym

    # rand_agent = RandomAgent(
    #     action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    class my_agent:
        def __init__(self):

            # self.action_space = action_space
            self.frame_stack = None
            self.frame_size = 8

            # 加載訓練好的代理模型
            # model_path = "log/best_model_austria.zip" # austria OK 2.5 OK 1.4 3.1
            model_path = "log/best_model_circle.zip" # circle OK 1.55


            print("Loading model will take some time...")
            self.model = PPO.load(model_path)
            print("Model loaded!")

        def preprocess_observation(self, obs):
            import cv2
            obs = obs.transpose(1, 2, 0)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            if self.frame_stack is None:
                self.frame_stack = np.stack([obs] * self.frame_size, axis=2)
            else:
                self.frame_stack = np.roll(self.frame_stack, -1, axis=2)
                self.frame_stack[:, :, -1] = obs
            return self.frame_stack

        def act(self, observation):
            observation = self.preprocess_observation(observation)
            action, _ = self.model.predict(observation, deterministic=True)
            # print("action shape is",action.shape)

            return action
        
    agent_instance = my_agent()
    
    connect(agent_instance, url=args.url)

'''
#server
python3 server.py --port 33333 --sid 313552041 --scenario austria_competition

#client
python my_client.py --url http://192.168.2.167:33333
'''

