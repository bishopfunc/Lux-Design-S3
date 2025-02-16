from stable_baselines3 import PPO
from ppo_game_env import PPOGameEnv
import os 

# 创建环境实例
env = PPOGameEnv()

# 使用多层感知机策略初始化 PPO 模型
model = PPO("MultiInputPolicy", env, verbose=1)

# 训练 10000 个时间步（可根据需要调整）
model.learn(total_timesteps=1000000, progress_bar=True)

# 保存训练好的模型
if os.path.exists("/kaggle"):
    model.save("/kaggle/working/agent/ppo_game_env_model")
else:
    model.save("ppo_game_env_model")

# 测试：加载模型并进行一次模拟
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
