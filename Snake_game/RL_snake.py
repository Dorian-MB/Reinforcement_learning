import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# The make_vec_env function from Stable Baselines 3 is used to create vectorized environments. 
# Vectorized environments allow you to run multiple instances of an environment in parallel, 
# providing a more efficient way to collect experiences (states, actions, rewards, etc.) during training.


class LinearQNet(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=32):
        super(LinearQNet, self).__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            n_flatten = self.flatten(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim),
                nn.ReLU(),
        )

    def forward(self, X):
        flat = self.flatten(X)
        out = self.linear(flat)
        return out

    
def evaluate_model(model, eval_env, num_episodes=10):
    all_rewards = []
    for episode in range(num_episodes):
        print(f"evaluation {episode=}")
        obs = eval_env.reset()
        done = False
        total_rewards = 0
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_rewards += reward
            if done :
                break
            
        all_rewards.append(total_rewards)
    average_reward = sum(all_rewards) / num_episodes
    return average_reward

