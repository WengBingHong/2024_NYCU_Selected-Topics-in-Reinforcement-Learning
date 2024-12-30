import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from racecar_gym.env_circle_gray_resize import RaceEnv

import gymnasium as gym

# 設置日誌路徑
log_path = "./log/train7/train_PPO_7/"

# circle_cw_competition_collisionStop
# austria_competition_collisionStop
# austria_competition
# 設定賽道名稱
map_name = "circle_cw_competition_collisionStop"


# 創建多個環境
def make_env(map, rank):
    """
    創建單一環境，使用 Monitor 包裝。
    map: 賽道名稱
    rank: 環境的標識符，用於區分不同環境
    """

    def _init():
        reset_when_collision = map.startswith("austria")  # 判斷是否以 "austria" 開頭
        # reset_when_collision = False
        env = RaceEnv(
            scenario=map,
            render_mode="rgb_array_birds_eye",
            reset_when_collision=reset_when_collision,  # 適用於特定賽道
        )
        return env
        # return Monitor(env, filename=f"{log_path}/monitor_env_{rank}.log")

    return _init

def linear_schedule(progress_remaining):
    """
    progress_remaining: 表示剩余训练比例，从 1.0 (开始) 到 0.0 (结束)
    返回学习率
    """
    initial_lr = 3e-4
    final_lr = 1e-5
    return final_lr + progress_remaining * (initial_lr - final_lr)

if __name__ == "__main__":
    # 使用 num_envs 個並行環境
    num_envs = 12
    envs = SubprocVecEnv([make_env(map_name, i) for i in range(num_envs)])

    # 添加 FrameStack 功能（堆疊 8 幀）
    stacks = 8
    envs = VecFrameStack(envs, n_stack=stacks, channels_order="first")
    envs = VecMonitor(envs)

    # 創建單一評估環境
    eval_env = SubprocVecEnv([make_env(map_name, i) for i in range(1)])
    eval_env = VecFrameStack(
        eval_env, n_stack=stacks, channels_order="first"
    )  # 評估環境也需要幀堆疊
    eval_env = VecMonitor(eval_env)

    # 初始化 PPO 模型
    model = PPO(
        "CnnPolicy",
        envs,  # 使用帶幀堆疊的環境
        # learning_rate=3e-4,
        # learning_rate=1e-4,
        # learning_rate=3e-5, # 調低
        # learning_rate = 1e-5, # 再調低
        learning_rate = linear_schedule,
        n_steps=1024,
        batch_size=64,
        # gamma=0.99,
        # gae_lambda=0.95,
        clip_range=0.2,
        # ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        n_epochs=10,
        use_sde=True,
        verbose=1,
        device="cuda",
        tensorboard_log=log_path,
    )

    # 設置評估回調
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # 加載預訓練模型
    pretrained_model = PPO.load("log/train7/train_PPO_7/PPO_12/best_model_circle.zip")
    
    # 加載預訓練模型的策略權重
    model.policy.load_state_dict(pretrained_model.policy.state_dict())

    # 開始訓練
    model.learn(
        total_timesteps=3e6, log_interval=10, callback=eval_callback, progress_bar=True
    )

    # 保存模型
    model.save(log_path + "ppo_model_" + map_name + "/")
