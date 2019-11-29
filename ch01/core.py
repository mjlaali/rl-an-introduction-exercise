from abc import abstractmethod
import numpy as np

import tqdm


def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)


def evaluate(bandit_env, algorithm, metrics, num_steps, num_evaluations):
    for _ in tqdm.tqdm(range(num_evaluations)):
        observation = bandit_env.reset()
        metrics.begin(algorithm)
        for step in range(num_steps):
            action = algorithm.get_action(observation)
            new_observation, reward, is_done, info = bandit_env.step(action)
            algorithm.update(observation, action, reward)
            metrics.update(step, action, reward, info)
            if is_done:
                break

        metrics.end()


class MetricEvaluator(object):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def begin(self, algorithm):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def update(self, step, action, reward, info):
        pass

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def value(self):
        pass

    @property
    def summary(self):
        return {self.name: self.value}


class RewardEvaluator(MetricEvaluator):
    def __init__(self):
        super(RewardEvaluator, self).__init__("avg_reward")
        self._reward_sum = None
        self._cur_exp_rewards = None
        self._num_exp = 0

    def begin(self, algorithm):
        self._cur_exp_rewards = list()

    def end(self):
        if self._reward_sum is None:
            self._reward_sum = np.asarray(self._cur_exp_rewards)
        else:
            self._reward_sum += np.asarray(self._cur_exp_rewards)
        self._num_exp += 1

    def update(self, step, action, reward, info):
        assert step == len(self._cur_exp_rewards), f"Out of sync, {step} != {len(self._cur_exp_rewards)}"
        self._cur_exp_rewards.append(reward)

    @property
    def value(self):
        return self._reward_sum / self._num_exp


class Policy(object):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def get_action(self, observation):
        pass

    @abstractmethod
    def update(self, observation, action, reward):
        pass

    @property
    def name(self):
        return self._name
