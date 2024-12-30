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
            # 加載訓練好的代理模型
            print("Loading model will take some time...")

            # self.model = DQN.load("log/train1/train_DQN_1/dqn_model_circle_cw_competition_collisionStop.zip") # 失敗?
            self.model = PPO.load("log/train5/train_PPO_5/best_model.zip")
            # self.model = TD3.load("log/train1/train_TD3_1/td3_model_circle_cw_competition_collisionStop.zip")
            # self.model = SAC.load("log/train1/train_SAC_1/sac_model_circle_cw_competition_collisionStop.zip")
            
            print("Model loaded!")

            self.is_Discrete = False

            # 檢查動作空間是否為 Discrete 類型
            if isinstance(self.model.action_space, gym.spaces.Discrete):
                print("動作空間是 Discrete 類型", type(self.model.action_space))
                
                self.is_Discrete = True

            else:
                print(f"動作空間不是 Discrete,而是 {type(self.model.action_space)}")
                self.is_Discrete = False

        def act(self, observation):
            
            print(self.is_Discrete)
            action, _ = self.model.predict(observation, deterministic=True)
            print("action shape is",action.shape)
            
            # if(self.is_Discrete):
            #     print("===== discrete")
            #     motor, steer = action  # 根據索引獲取對應的 (motor, steer)
            #     action = np.array([motor, steer], dtype=np.float32)

            return action
        
    agent_instance = my_agent()
    
    connect(agent_instance, url=args.url)

'''
#server
python3 server.py --port 33333 --sid 313552041 --scenario austria_competition
python3 server.py --port 3583 --sid 313552041 --scenario austria_competition # fail

#client
ssh hrs@140.113.213.49 -p 2280
python my_client.py --url http://192.168.2.167:33333
python client.py --url http://140.113.213.49:3583 # fail
'''

