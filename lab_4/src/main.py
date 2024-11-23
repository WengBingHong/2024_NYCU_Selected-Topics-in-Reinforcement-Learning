from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_test_5_ou_noise_delay_4_new_reward/',
		"update_freq": 4,
		"eval_interval": 10,
		"eval_episode": 10,
		"clip_c": 0.5, # clipping threshold for the noise
		"sigma_target": 0.2, # Standard deviation of target smoothing noise
	}
	agent = CarRacingTD3Agent(config)
	agent.train()


