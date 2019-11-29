import math

import numpy as np

from ch01.bandit import RandomPolicy, Bandit
from ch01.core import evaluate, RewardEvaluator


def test_random_policy():
    num_arms = 10
    env = Bandit(num_arms)

    policy = RandomPolicy(env.action_space.n)
    metric = RewardEvaluator()
    num_steps = 100000
    evaluate(env, policy, metric, num_steps, 1)
    avg_reward = np.mean(metric.value)
    # 0.999 confidence interval = 3.090 * sigma ^ 2 / sqrt(n)
    # 0.95 confidence interval = 1.645 * sigma ^ 2 / sqrt(n)
    decimal = -int(math.log10(3.090 * num_arms / math.sqrt(num_steps)))
    assert decimal >= 1
    np.testing.assert_almost_equal(avg_reward, np.mean(env._reward_mean), decimal=decimal)
