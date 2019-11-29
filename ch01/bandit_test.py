import math

import numpy as np
import pytest

from ch01.bandit import RandomPolicy, Bandit
from ch01.core import evaluate, RewardEvaluator


@pytest.mark.parametrize("num_eval", [1, 2])
def test_random_policy(num_eval):
    num_arms = 10
    arms_mean_reward = []

    def reset_listener(reward_mean):
        arms_mean_reward.append(reward_mean)

    env = Bandit(num_arms, reset_listener)

    policy = RandomPolicy(env.action_space.n)
    metric = RewardEvaluator()
    num_steps = 100000
    evaluate(env, policy, metric, num_steps, num_eval)
    avg_reward = np.mean(metric.value)
    # 0.999 confidence interval = 3.090 * sigma ^ 2 / sqrt(n)
    # 0.95 confidence interval = 1.645 * sigma ^ 2 / sqrt(n)
    decimal = -int(math.log10(1.645 * num_arms * num_eval / math.sqrt(num_steps * num_eval)))

    expected_reward = np.mean(arms_mean_reward)
    assert decimal >= 1
    np.testing.assert_almost_equal(avg_reward, expected_reward, decimal=decimal)


