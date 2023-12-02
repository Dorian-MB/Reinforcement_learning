import gym
from gym import spaces
import numpy as np
from snake_ui import SnakeGame

from constant import *

MAX_LENGHT = 10

class SnakeEnv(gym.Env):
    """
    step(action): This method takes an action as input, updates the game state based on that action, returns the new state, the reward gained (or lost), whether the game is over (done), and additional info if necessary.
    reset(): This method resets the environment to an initial state and returns this initial state. It's used at the beginning of a new episode.
    render(): This method is for visualizing the state of the environment. Depending on how you want to view the game, this could simply update the game window.
    close(): This method performs any necessary cleanup, like closing the game window.
    """
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4) # Output
        self.observation_space = gym.spaces.Box(low=-N, high=N,
                                            shape=(13,), dtype=np.float64)
        self.snake_game = None
        self.previous_score = 0
        self._last_distance = 0
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = {"renders_mode":["human"]}
        
    def seed(self, seed=42): # needed with make_vec_env
        np.random.seed(seed)
    
    def get_snake_and_food_position(self, raw_obs):
        food_position = np.argwhere(raw_obs == 2).flatten()
        snake_positions = np.argwhere(raw_obs == 1).flatten()
        return snake_positions, food_position
        
    def euclidean_distance_centroid(self, raw_obs):
        snake_positions, food_position = self.get_snake_and_food_position(raw_obs)
        snake_centroid = np.mean(snake_positions)

        new_distance = np.linalg.norm(snake_centroid - food_position)
        return new_distance
    
    def feature_gen(self, raw_obs):
        new_distance = self.euclidean_distance_centroid(raw_obs)
        delta_distance = np.array([self._last_distance - new_distance])
        
        snake_positions, food_positions = self.get_snake_and_food_position(raw_obs)
        delta_lenght = max(MAX_LENGHT - len(snake_positions), 0)
        snake_positions = np.pad(snake_positions, (0, delta_lenght), mode='constant')

        obs = np.concatenate([snake_positions.flatten(), food_positions.flatten(), delta_distance])
        return obs
        
    def get_reward(self, raw_obs, score, done):
        # Calculate the Euclidean distance between the snake and the food
        new_distance = self.euclidean_distance_centroid(raw_obs)
        # Check if the snake has eaten food and update the reward
        if self.previous_score != score:
            reward = 100
            self.previous_score = score
        elif done:
            reward = -10
        else:
            reward =  1/2 if new_distance < self._last_distance else -1/10
        self._last_distance = new_distance
        return reward

    def step(self, action):
        raw_obs, score, done, _ = self.snake_game.step(action)
        obs = self.feature_gen(raw_obs)
        reward = self.get_reward(raw_obs, score, done)
        return obs, reward, done, _

    def reset(self):
        self.snake_game = SnakeGame()
        raw_obs = self.snake_game.raw_obs
        obs = self.feature_gen(raw_obs)
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            self.snake_game.render()
            
    def close(self):
        self.snake_game.quit()

if __name__ =="__main__":
    env = SnakeEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # choose a random action
        obs, reward, done, info = env.step(action)
        env.render()

    env.close()
