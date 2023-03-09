import debugpy
import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v2")

# Instantiate the agent
#model = DQN("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
#model.learn(total_timesteps=int(2e5), progress_bar=True)
# Save the agent
#model.save("dqn_lunar")
print('model saved')
#del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

#from huggingface_hub import notebook_login
#notebook_login()

""" push_to_hub(
    repo_id="ThomasSimonini/ppo-LunarLander-v2",
    filename="ppo-LunarLander-v2.zip",
    commit_message="Added LunarLander-v2 model trained with PPO",
) """