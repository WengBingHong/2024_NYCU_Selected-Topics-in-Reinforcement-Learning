from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/Enduro_release_3/',
		"update_ppo_epoch": 3,
		# "learning_rate": 2.5e-4,
		# "learning_rate": 2.5e-5,
		"learning_rate": 2.5e-6,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 100,
		"eval_episode": 3,
	}
	agent = AtariPPOAgent(config)
	agent.load("log/Enduro_release_1/model_69747642_1530.pth")
	agent.train()



