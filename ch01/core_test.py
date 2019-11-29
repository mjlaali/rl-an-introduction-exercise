import unittest
from unittest.mock import MagicMock
import numpy as np

from ch01.core import RewardEvaluator


class TestRewardEvaluator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRewardEvaluator, self).__init__(*args, **kwargs)
        self._evaluator = RewardEvaluator()

    def test_score(self):
        algorithm = MagicMock()
        for exp_idx in range(2):
            self._evaluator.begin(algorithm)
            for step in range(5):
                self._evaluator.update(step, MagicMock(), exp_idx + step, MagicMock())
            self._evaluator.end()

        np.testing.assert_almost_equal(self._evaluator.value, np.asarray(range(5)) + 0.5)