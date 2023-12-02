from stable_baselines3 import PPO,DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn

from snake_env import SnakeEnv
from RL_snake import LinearQNet, evaluate_model

train_env = make_vec_env(lambda: SnakeEnv(), n_envs=2)
eval_env = SnakeEnv()


policy_kwargs = dict(
    features_extractor_class=LinearQNet,
)

learning_rate_schedule = get_schedule_fn(0.0003)
model = PPO("CnnPolicy", train_env, policy_kwargs=policy_kwargs, verbose=2)

new_logger = configure("path_to_save_logs", ["stdout", "tensorboard"])
model.set_logger(new_logger) # Run TensorBoard in a terminal: tensorboard --logdir=path_to_save_logs


total_timesteps = 200_000
eval_interval = 50_000  
num_eval_episodes = 10  

# Training loop with periodic evaluation
for i in range(0, total_timesteps, eval_interval):
    print(f"Training session N°{i/eval_interval} {'-'*50}")
    model.learn(total_timesteps=eval_interval)
    avg_reward = evaluate_model(model, eval_env, num_episodes=num_eval_episodes)
    print(f"Evaluation average reward: {avg_reward}")

model.save("ppo_snake")


"""
POO returns :
    - ep_len_mean :             Average number of steps per episode.
    - ep_rew_mean :             Average reward earned per episode.
    - fps :                     Number of environment steps processed per second (speed of the simulation).
    - iterations :              Number of batches of data processed.
    - time_elapsed:             Total training time in seconds.
    - total_timesteps :         Total number of environment steps experienced by the agent.
    - approx_kl :               Measure of policy change after an update.
    - clip_fraction :           Proportion of time clipping is used in PPO. 
    - clip_range :              Range for policy update clipping in PPO.
    - entropy_loss :            Indicates exploration level (higher is more exploration).
    - explained_variance :      How well the value function predicts return.
    - learning_rate :           Current learning rate for optimization.
    - loss :                    Total combined loss being optimized.
    - n_updates :               Number of updates to the model so far.
    - policy_gradient_loss :    Loss from the policy gradient update.
    - value_loss :              Loss in predicting expected returns.
"""




