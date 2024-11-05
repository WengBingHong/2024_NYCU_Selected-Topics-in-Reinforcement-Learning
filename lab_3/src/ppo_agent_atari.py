import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
    def __init__(self, config):
        super(AtariPPOAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        # self.env = ???
        self.env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
        self.env.metadata["render_fps"] = 30  # Set to your desired FPS, e.g., 30
        # self.env = gym.wrappers.RecordVideo(self.env, "env_video")
        self.env = gym.wrappers.AtariPreprocessing(
            self.env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        self.env = gym.wrappers.FrameStack(self.env, 4)
        ### TODO ###
        # initialize test_env
        # self.test_env = ???
        self.test_env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
        self.test_env.metadata["render_fps"] = 30  # Set to your desired FPS, e.g., 30
        # self.test_env = gym.wrappers.RecordVideo(self.test_env, "test_env_video")
        self.test_env = gym.wrappers.AtariPreprocessing(
            self.test_env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        self.test_env = gym.wrappers.FrameStack(self.test_env, 4)

        # load model
        # self.load_and_evaluate('lab_3_2/Code/log/ref_model/model_82659120_1350.pth')

        self.net = AtariNet(self.env.action_space.n)
        self.net.to(self.device)
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def decide_agent_actions(self, observation, eval=False):
        ### TODO ###
        # add batch dimension in observation

        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        # print(observation.shape)
        # print(len(observation.shape))
        observation = torch.from_numpy(observation)
        observation = observation.to(
            self.device, dtype=torch.float32
        )  # Move to device and set dtype
        # print(observation.shape)
        # print(len(observation.shape))

        # get action, value, logp from net
        # if eval:
        #     with torch.no_grad():
        #         ???, ???, ???, _ = self.net(observation, eval=True)
        # else:
        #     ???, ???, ???, _ = self.net(observation)
        if eval:
            with torch.no_grad():
                action, logp_pi, value, _ = self.net(observation, eval=True)
        else:
            action, logp_pi, value, _ = self.net(observation)

        action = action.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        logp_pi = logp_pi.cpu().detach().numpy()
        
        # print(action.shape)
        # print(value.shape)
        # print(logp_pi.shape)

        return action, value, logp_pi

    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(
            self.discount_factor_gamma, self.discount_factor_lambda
        )
        sample_count = len(batches["action"])
        batch_index = np.random.permutation(sample_count)

        observation_batch = {}
        for key in batches["observation"]:
            observation_batch[key] = batches["observation"][key][batch_index]
        action_batch = batches["action"][batch_index]
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]

        for _ in range(self.update_count):
            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][
                        start : start + self.batch_size
                    ]
                ac_train_batch = action_batch[start : start + self.batch_size]
                return_train_batch = return_batch[start : start + self.batch_size]
                adv_train_batch = adv_batch[start : start + self.batch_size]
                v_train_batch = v_batch[start : start + self.batch_size]
                logp_pi_train_batch = logp_pi_batch[start : start + self.batch_size]

                ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
                ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
                ac_train_batch = torch.from_numpy(ac_train_batch)
                ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(
                    self.device, dtype=torch.float32
                )
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(
                    self.device, dtype=torch.float32
                )

                ### TODO ###
                # calculate loss and update network
                # ???, ???, ???, ??? = self.net(...)
                _, log_prob, value, entropy = self.net(
                    ob_train_batch, False, torch.squeeze(ac_train_batch)
                )

                # calculate policy loss
                # ratio = ???
                # surrogate_loss = ???
                ratio = torch.exp(log_prob - logp_pi_train_batch)
                surrogate_1 = ratio * adv_train_batch
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * adv_train_batch
                )
                surrogate_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))
                # print(surrogate_loss)

                # calculate value loss
                # value_criterion = nn.MSELoss()
                # v_loss = value_criterion(...)
                value_criterion = nn.MSELoss()
                v_loss = value_criterion(value, return_train_batch)

                # calculate total loss
                # loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
                loss = (
                    surrogate_loss
                    + self.value_coefficient * v_loss
                    - self.entropy_coefficient * entropy
                )
                loss = loss.mean()

                # update network
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
                self.optim.step()

                # print(surrogate_loss)
                total_surrogate_loss += surrogate_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                loss_counter += 1

        self.writer.add_scalar(
            "PPO/Loss", total_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Surrogate Loss",
            total_surrogate_loss / loss_counter,
            self.total_time_step,
        )
        self.writer.add_scalar(
            "PPO/Value Loss", total_v_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Entropy", total_entropy / loss_counter, self.total_time_step
        )
        print(
            f"Loss: {total_loss / loss_counter}\
            \tSurrogate Loss: {total_surrogate_loss / loss_counter}\
            \tValue Loss: {total_v_loss / loss_counter}\
            \tEntropy: {total_entropy / loss_counter}\
            "
        )
