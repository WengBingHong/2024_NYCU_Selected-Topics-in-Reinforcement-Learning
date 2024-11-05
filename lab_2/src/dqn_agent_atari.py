import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
from gym.wrappers import FrameStack
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		#, mode = 'rgb_array'
		
		# this is Pacman
		self.env = gym.make('ALE/MsPacman-v5', render_mode = 'rgb_array')
		self.env.metadata['render_fps'] = 30  # Set to your desired FPS, e.g., 30
		self.env = gym.wrappers.RecordVideo(self.env, 'env_video')
		self.env = gym.wrappers.AtariPreprocessing(self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.env = gym.wrappers.FrameStack(self.env, 4)
		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make('ALE/MsPacman-v5', render_mode = 'rgb_array')
		self.test_env.metadata['render_fps'] = 30 # Set to your desired FPS, e.g., 30
		self.test_env = gym.wrappers.RecordVideo(self.test_env, 'test_env_video')
		self.test_env = gym.wrappers.AtariPreprocessing(self.test_env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)
	
		# this is Enduro
		'''	
		self.env = gym.make('ALE/Enduro-v5', render_mode = 'rgb_array')
		self.env.metadata['render_fps'] = 30  # Set to your desired FPS, e.g., 30
		# self.env = gym.wrappers.RecordVideo(self.env, 'env_video')
		self.env = gym.wrappers.AtariPreprocessing(self.env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.env = gym.wrappers.FrameStack(self.env, 4)
		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make('ALE/Enduro-v5', render_mode = 'rgb_array')
		self.test_env.metadata['render_fps'] = 30 # Set to your desired FPS, e.g., 30
		# self.test_env = gym.wrappers.RecordVideo(self.test_env, 'test_env_video')
		self.test_env = gym.wrappers.AtariPreprocessing(self.test_env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)
		'''
	
		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)

		# load model
		# self.load_and_evaluate('log/model_19993117_4664.pth')

		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		if np.random.random() < epsilon:
			# Exploration: Select a random action, generates a random action from the environment’s action space.
			action = action_space.sample()
		else:
			# Exploitation: Use the behavior network to select the best action
			array_observation = np.array(observation)
			# Convert the observation to a PyTorch tensor and add a batch dimension
			tensor_observation = torch.FloatTensor(array_observation).unsqueeze(0).to(self.device)
			
			# Use the behavior network to predict the Q-values for each action
			with torch.no_grad(): # disable gradient calculation
				# Forward pass to get Q-values
				action_val = self.behavior_net(tensor_observation)
				# Select the action with the highest Q-value
				action = torch.max(action_val, 1)[1].item()
				
		return action

	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		# 1. Get the Q-values from the behavior network for the actions taken	
		# q_value = self.behavior_net(state).gather(1, action.long()) # forward # action is of type long for indexing
		q_values = self.behavior_net(state)
		q_value = q_values.gather(1, action.long())

		# 2. choose DQN, DDQN or Dueling

		# 2.1 This is DQN implementation
		'''
		# Compute the target Q-values using the target network
		with torch.no_grad():
			# Get the maximum Q-value for the next state from the target network
			q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
		'''
		# DQN

		# 2.2 This is DDQN implementation
		with torch.no_grad():
			# Select the best action using the behavior network
			next_best_actions = self.behavior_net(next_state)
			next_best_action = next_best_actions.argmax(1).unsqueeze(1)

			# Use the target network to evaluate the Q-value of the selected action
			q_next = self.target_net(next_state).gather(1, next_best_action)
			
		# DDQN

		# 3. Bellman equation: Q_target = reward + gamma * max(Q(s', a)) for non-terminal states
		# (done = 1 if the episode is over, otherwise done = 0)
		q_target = torch.where(done.bool(), reward, reward + self.gamma * q_next)

		# 4. Calculate the loss between the current Q-values and the target Q-values
		criterion = torch.nn.MSELoss() # init Mean Squared Error (MSE) Loss function
		loss = criterion(q_value, q_target)

		# Optionally, log the loss for monitoring purposes (e.g., in TensorBoard)
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
	
		# 5. Backpropagate the loss and update the behavior network
		self.optim.zero_grad() # Reset the gradients. Gradients need to be reset before each backpropagation step
		loss.backward() # Compute the gradients. performs backpropagation
		self.optim.step() # Update the behavior network weights. updates the model's weights using the gradients computed during backpropagation

		
