from ch01.core import Policy
import gym
from gym import spaces
import random
import numpy as np


class Bandit(gym.Env):
    def __init__(self, num_arms, reset_listener=None):
        self._num_arms = num_arms
        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)
        self._reward_mean = None
        self._reset_listener = reset_listener

    def step(self, action):
        assert self.action_space.contains(action)
        return 0, self._reward_mean[action] + np.random.normal(0, 1), False, \
               {'num_arms': self.num_arms, 'reward_mean': self._reward_mean}

    def reset(self):
        self._reward_mean = np.random.normal(0, 1, (self.num_arms, ))
        if self._reset_listener:
            self._reset_listener(self._reward_mean)

    def render(self, mode='human'):
        pass

    @property
    def num_arms(self):
        return self._num_arms


class RandomPolicy(Policy):
    def __init__(self, num_actions):
        super(RandomPolicy, self).__init__("random_policy")
        self._num_actions = num_actions

    def get_action(self, observation):
        return random.randint(0, self._num_actions - 1)

    def update(self, observation, action, reward):
        pass